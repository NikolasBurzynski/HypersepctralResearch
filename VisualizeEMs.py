import numpy as np
# from PIL import Image
import scipy.misc as smp
import math
import tifffile as tf
import glob


IM_SIZE = 1024

def main():
    files = []
    for file in glob.glob("sciOptimizeRes/" + "*.tif.npy"):
        files.append(file.split("\\")[-1])
    print(files)
    selectedFile = int(input("Which unmixing result would you like to visualize"))    
    root_file = str("sciOptimizeRes/" + files[selectedFile])[:-4]
    coeffs = np.load(root_file + ".npy")
    print(coeffs.shape)
    funs = np.load(root_file + "_FUNS.npy")
    # print(funs.shape)
    image = np.delete(coeffs, 3, 1).reshape(IM_SIZE,IM_SIZE,3)
    splits = np.array(np.hsplit(coeffs, 4))
    #BSA, DNA, OA
    # splits[0] = splits[0] * 1902.5469
    # splits[1] = splits[1] * 419.669435
    # splits[2] = splits[2] * 9525.879
    print(splits)
    BSA = splits[2].reshape((IM_SIZE,IM_SIZE))
    DNA = splits[0].reshape((IM_SIZE,IM_SIZE))
    OA = splits[1].reshape((IM_SIZE,IM_SIZE))
    WATER = splits[3].reshape((IM_SIZE,IM_SIZE))
    # print(funs)
    tf.imwrite(root_file + "color.tif", image, photometric = 'rgb')
    tf.imwrite(root_file + "error.tif", funs, photometric = "minisblack")


    tf.imwrite(root_file + "BSA_EM.tif", BSA, photometric = "minisblack")
    tf.imwrite(root_file + "DNA_EM.tif", DNA, photometric = "minisblack")
    tf.imwrite(root_file + "OA_EM.tif", OA, photometric = "minisblack")
    tf.imwrite(root_file + "WATER_EM.tif", WATER, photometric = "minisblack")
    


if __name__ == "__main__":
    main()