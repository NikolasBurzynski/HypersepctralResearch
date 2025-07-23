from cv2 import cartToPolar
from skimage import io
from skimage.filters import threshold_otsu
import glob
import cv2
import numpy as np
from torch.utils.data import Dataset
import math
from PIL import Image
from scipy import signal
import matplotlib.pyplot as plt
import tifffile as tf
from dataBalancer import equalizeData
from scipy.interpolate import interp1d
import random


'''
Data.py

This module contains the Data class which inherits the Dataset class from PyTorch. 
This class is responsible for providing the test and training data. 

__init__: The init function of this class takes 2 required parameters and 1 optional parameter
            1. path: path to the folder with the HeLa cell tifs inside of it
            2. normed: a boolean that determines if the data will be normalized between 1 and 0
                - We normalize the data because the endmembers are normalized so the network outputs will be small coefficients
            3. labeled (optional): We played around with labeled data, if this argument is set to True, the code will expect png labels
                                   to be present in order to add a label layer to the training data (not really using that much atm)

        The init function first uses the "glob" module to go through all of the files in the "path" with .tif extensions
        It then opens these images, and put them, as well as their names into lists. It them prompts you to choose which file you want
        to use as your training data and which for testing. Finally it takes your selections, sanatizes the data via doClean() and then
        does the normalization if normed is true.

getImage(): This just returns the training image data which is used in the SciPy unmixing in unmixing.py
'''

class Data(Dataset):

    def process_image(self, filename):
        image = io.imread(filename, plugin='tifffile')
        image = np.array(image, dtype=np.float32)
        image = np.moveaxis(image, 0, -1) #Important for making the shape x, y, z
        return image

    def __init__(self, path, lsqe_mode = False):
        self.path = path
        self.slice_filenames = []
        for i, file in enumerate(glob.glob(path + "*.tif")):
            self.slice_filenames.append((i, file))

        if not lsqe_mode:
            print("All Files: " , self.slice_filenames)

            """
                Two sets of data. Train-time data and testing data. The train-time data will be split between training and validation sets and the testing data will just be the test
                data. The train-time data needs to be labeled and have lipid droplets to it. The testing data shouldn't have any of these things applied. 

                

                Each of the HELA tif files include a nuclei label slice at the beginning. These will be useful in the train-time sets but not in the test sets, we will deal with them
                at the use site.

                I think a good idea would be to have the user pick what they want the test file to be and then just assume that the rest of the files will be used for the train-time set
            """


            testFile = int(input("Which file is Test Data?"))
            self.test_file_index = testFile
            self.testDataName = self.slice_filenames[testFile][1]
            self.two_d_test_data = self.process_image(self.testDataName).astype(np.float32)
            self.testDataShape = (self.two_d_test_data.shape[0], self.two_d_test_data.shape[1])
            self.testData = self.flatten_norm_and_clean(self.two_d_test_data)

            self.training_set = []
            # Now that we have the test_data we can assume that the rest of the files should be part of the training data. 
            for (i, file_name) in self.slice_filenames:
                if i == testFile: # Skip this file since this is the test file that we chose
                    continue
                '''
                What are the steps for preparing the train-time set?
                    1. We need to run thresholding on the nuclei layer to get it to be either 0 or 1
                    2. Use the lipid and nuclei labels to replace some non nuclei pixels with lipid droplet pixels
                '''
                img = self.process_image(file_name)
                img = self.flatten_norm_and_clean(img)

                nuclei_slice = img[:, 1]
                
                threshold = threshold_otsu(nuclei_slice) # Find the threshold value of the nuclei label layer
                print(f"Found Otsu's threshold of {threshold}")
                print("Applying the threshold")
                img[:, 1] = [0 if x < threshold else 1 for x in nuclei_slice]
                threshold_image = np.reshape(np.array(img[:, 1]), (1024,1024))
                print(np.amax(threshold_image))
                plt.imshow(threshold_image, cmap='gray')
                plt.title(f"BINARY THRESHOLD FOR {file_name}")
                plt.show()
                print(img.shape)
                
                img = self.add_lipid_droplets(img)

                self.training_set.extend(img)
            
            self.training_set = np.array(self.training_set)
            

            
            
    
    def add_lipid_droplets(self, img):
        # This function takes in a set of pixels from one of the train time set stacks and the lipid label array. It returns a copy of the img with 30% of the pixels being
        # lipid droplets
        print("Getting indexes of Lipid Droplets")
        lipid_slice = img[:, 0] # The first slice is the lipid label
        nuclei_slice = img[:, 1] # The second slice is the nuclei label
        actual_pixel_spectra = img[:, 2:] # The rest of the slice is the actual spectra
        idxs = [i for i in range(len(actual_pixel_spectra)) if lipid_slice[i] != 0]
        current_lipid_count = len(idxs)
        print(f"Found {current_lipid_count} lipid droplet pixels")
        target_lipid_count = math.floor((1.0 / 30.0) * len(actual_pixel_spectra))
        num_to_add = target_lipid_count - current_lipid_count
        print(f"Working on adding {num_to_add} lipid droplets to the training image so that there are {target_lipid_count} total")
        for _ in range(num_to_add):
            target_lipid_indx = random.choice(idxs) # Pick a random lipid-droplit pixel location
            dest_indx = random.choice(range(len(actual_pixel_spectra))) # Pick a random pixel location
            while(dest_indx in set(idxs) and nuclei_slice[dest_indx] != 1): # Keep picking till it is not a lipid droplet pixel and not a nuclei
                # Since we make sure that the destination pixel isn't a nuclei, we save ourselves from having to change the nuclei labels
                dest_indx = random.choice(range(len(actual_pixel_spectra)))
            img[:, 2:][dest_indx] = actual_pixel_spectra[target_lipid_indx] # Copy the target lipid droplet pixel to the new location
        print(f"Finished adding {num_to_add} lipid pixels to a train time stack")
        return img



    def flatten_norm_and_clean(self, img):
        flattened_img = np.reshape(img, (-1, img.shape[-1])) # Flatten slices into a multidimensional array
        flattened_img[flattened_img < 0] = 0
        flattened_img[flattened_img == np.inf] = 0
        flattened_img = np.nan_to_num(flattened_img)
        flattened_img[:, 2:] = [pixel - np.min(pixel) for pixel in flattened_img[:, 2:]] # Apply subtraction to the spectra part of the flat data. leaving the lipid and nuclei labels alone
        flattened_img[:, 2:] = (flattened_img[:, 2:] / np.max(flattened_img[:, 2:])) #* 255
        return flattened_img

    
    def getImage(self):
        return self.testData
    
    def getImages(self):
        images = []
        for (i, file_name) in self.slice_filenames:
            img = self.process_image(file_name)
            self.testDataShape = (img.shape[0], img.shape[1])
            img = self.flatten_norm_and_clean(img)
            images.append((img[:, 2:], file_name)) # Get rid of the first two slices since those are lipid and dna labels and we dont need them
        return images

