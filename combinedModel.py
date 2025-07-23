from NN import CombinedNet
from Data import Data
from ManualCombination import ManualCombination
import tifffile as tf
from pathlib import Path



import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import datetime
import time


TRAINING_PARAMS = {
    "LEARNING_RATE" : .0001,
    "NUM_EPOCHS" : 30,
    "BATCH_SIZE" : 100,
}
WAVE_NUMBERS = 17
INPUT_LAYER_SIZE = WAVE_NUMBERS
DNA_FINDER_OUTPUT_SIZE = 1
UNMIXER_OUTPUT_SIZE = 3
BACK_BONE_OUTPUT_SIZE = 128
BACK_BONE_LAYER_SIZES = [INPUT_LAYER_SIZE, 32, 64, BACK_BONE_OUTPUT_SIZE]
DNA_FINDER_LAYER_SIZES = [BACK_BONE_OUTPUT_SIZE, 64, 32, 16, 8, DNA_FINDER_OUTPUT_SIZE]
UNMIXER_LAYER_SIZES = [BACK_BONE_OUTPUT_SIZE, 64, 32, 16, 8, UNMIXER_OUTPUT_SIZE]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
designMatrix = torch.from_numpy(np.load("Data_Files/17-withoutDNA-DesignMatrix_All_1_Norm_background_plus_water.npy")).to(device=device, dtype = torch.float)
designMatrix = designMatrix.permute(1,0)

DNA_PROB_THRESH = .60

#                       R,   G,   B     NULL
#Output activations are DNA PROB, OA,  BSA    Background


def mse_loss(inputs, outputs):
    resSpec = torch.mm(outputs, designMatrix)
    loss_type = torch.nn.MSELoss()
    return  loss_type(resSpec, inputs)


def calc_dna_loss(loss_fn, outputs, labels):

    outputs = outputs.squeeze()
    return loss_fn(outputs, labels)


def makeRGB(net, test_data, test_data_name, test_data_shape, device, MODEL_PATH, MODEL_IDENTIFIER):

    test_data = test_data[:, 2:WAVE_NUMBERS+2] # Get rid of the Nuclei and Lipid labeling layers
    test_data_name = test_data_name.split('_')[2]
    print(test_data_name)
    print(test_data_shape)

    testloader = torch.utils.data.DataLoader(test_data, batch_size = test_data_shape[0], shuffle = False, num_workers = 1, persistent_workers = True)
    
    checkpoint = torch.load(MODEL_PATH)
    net.load_state_dict(checkpoint["Model_State_Dict"])
    net.eval() # Turns off dropout
    image = np.zeros((test_data_shape[0], test_data_shape[1], 4)) # This is 4 since DNA is no longer part of the unmixing
    start = time.time()
    for i, data in enumerate(testloader, 0):
        inputs = data.to(device=device, dtype=torch.float)
        dna_out, unmixer_out = net(inputs)
        dna_out = dna_out.cpu().detach().numpy()
        unmixer_out = unmixer_out.cpu().detach().numpy() # OA, BSA, BACKGROUND
        combined = np.concatenate((dna_out, unmixer_out), axis = 1)
        image[i, :] = combined # Leave the 0 empty for "DNA"
    end = time.time()
    BG = image[:, :, -1]   # BACKGROUND
    dnaprob_oa_bsa = image[:, :, :-1] # DNA_prob, OA, BSA
    tf.imwrite("./Results/new/" + str(DNA_PROB_THRESH*100) + 'P' + MODEL_IDENTIFIER + "_" + test_data_name + "_raw_outputs.tif", dnaprob_oa_bsa, photometric = 'rgb')
    print("Created File")
    net.train()
    return dnaprob_oa_bsa, end-start
        

def train(net, train_time_set, device, MODEL_PATH, params, checkpoint = None, LOSSES = []):

    optimizer = torch.optim.AdamW(net.parameters(), lr = params["LEARNING_RATE"]) # AdamW correct implementation
    startEpoch = 0
    bce_loss = torch.nn.BCELoss()

    
    if checkpoint != None:
        checkpoint = torch.load(checkpoint)
        net.load_state_dict(checkpoint["Model_State_Dict"])
        optimizer.load_state_dict(checkpoint["Optimizer_State_Dict"])
        startEpoch = checkpoint["Epoch"]

    start = time.time()
    train_loader = DataLoader(train_time_set, batch_size = params["BATCH_SIZE"], shuffle = True, num_workers = 1, persistent_workers = True)
    for epoch in range(startEpoch, startEpoch + params["NUM_EPOCHS"]):
        running_loss = 0.0
        running_dna_loss = 0.0
        running_unmixing_loss = 0.0
        avg_loss = 0.0
        avg_dna_loss = 0.0
        avg_unmixing_loss = 0.0
        for i, data in enumerate(train_loader):
            nuclei_labels = data[:, 1].to(device=device, dtype=torch.float)
            spectra_inputs = data[:, 2:WAVE_NUMBERS+2].to(device=device, dtype=torch.float) # Get rid of the first two values per pixel, these are the lipid and nuclei labels
            optimizer.zero_grad()
            dna_finder_output, unmixer_outputs = net(spectra_inputs)
            dna_finder_loss = calc_dna_loss(bce_loss, dna_finder_output, nuclei_labels) * 0.1 # Weight this loss less because its value is 100 x the value of the unmixing loss
            unmixing_loss = mse_loss(spectra_inputs, unmixer_outputs)
            total_loss = dna_finder_loss + unmixing_loss
            total_loss.backward()
            running_loss += total_loss.item()
            running_dna_loss += dna_finder_loss.item()
            running_unmixing_loss += unmixing_loss.item()
            avg_loss = running_loss/(i+1)
            avg_dna_loss = running_dna_loss/(i+1)
            avg_unmixing_loss = running_unmixing_loss/(i+1)
            optimizer.step()
            print(f"AVG_Loss:{avg_loss:.8e} AVG_DNA_LOSS:{avg_dna_loss:.8e} AVG_UNMIXING_LOSS: {avg_unmixing_loss:.8e} Epoch: {epoch}\r", end = "")        
        print()
    end = time.time()
    save_path = MODEL_PATH + ".pth"
    torch.save({
        "Model_State_Dict": net.state_dict(),
        "Optimizer_State_Dict": optimizer.state_dict(),
        "Epoch": startEpoch + params["NUM_EPOCHS"]
    }, save_path)
    return save_path, end - start, LOSSES
            
if __name__ == "__main__":  
    data = Data("./Data_Files/HELA/")
    TEST_IMG_NAME = Path(data.slice_filenames[data.test_file_index][1]).name.split('.')[0]
    MODEL_IDENTIFIER = f"MSE_17WN_Train_1_2_3_{TRAINING_PARAMS['NUM_EPOCHS']}Epochs_{TRAINING_PARAMS['BATCH_SIZE']}Batch"
    MODEL_PATH = f'Models/{MODEL_IDENTIFIER}'
    print(f"MODEL PATH: {MODEL_PATH}")
    net = CombinedNet(BACK_BONE_LAYER_SIZES, UNMIXER_LAYER_SIZES, DNA_FINDER_LAYER_SIZES)
    device = torch.device("cuda:0")
    net.to(device)
    train_time_data = np.array(data.training_set)
    full_model, first_duration, LOSSES_1 = train(net, train_time_data, device, MODEL_PATH, TRAINING_PARAMS)

    # # print(f"TIME TO TRAIN MODEL: {first_duration}")

    # full_model = './Models/17WN_Train_1_2_3_30Epochs_100Batch.pth'
    test_data = np.array(data.testData)
    test_data_name = data.testDataName
    test_data_shape = data.testDataShape
    img, time_spent = makeRGB(net, test_data, test_data_name, test_data_shape, device, full_model, MODEL_IDENTIFIER)
    print(f"TIME SPENT CREATING OUTPUT IMAGE: {time_spent}")
    manualComb = ManualCombination(img=img, 
                                   designMatrix=designMatrix, 
                                   net=net, 
                                   testDataBundle=(test_data, test_data_name, test_data_shape), 
                                   device=device, 
                                   MODEL_PATH=full_model, 
                                   weighted=True, 
                                   weights=TRAINING_PARAMS["WEIGHTS"], 
                                   wave_numbers=WAVE_NUMBERS)
    manualComb.eventLoop()
