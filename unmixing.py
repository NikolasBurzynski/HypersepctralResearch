import numpy as np
from scipy.optimize import lsq_linear
import matplotlib.pyplot as plt
import os
from PIL import Image
from Data import Data
import time

IM_SIZE = 512


def main():
    factory = Data("./Data_Files/HELA/", lsqe_mode=True)
    designMatrix = np.load("Data_Files/21-withDNA-DesignMatrix_All_1_Norm_background_no_water.npy")
    doUnmixings(factory, designMatrix)
 


def doUnmixings(factory, designMatrix):
    plt.plot(designMatrix[:,0], label = "DNA", color = "red")
    plt.plot(designMatrix[:,1], label = "OA", color = "green")
    plt.plot(designMatrix[:,2], label = "BSA", color = "blue")
    plt.plot(designMatrix[:,3], label = "Background", color = "black")
    plt.legend()
    plt.show()

    images = factory.getImages()
    image_shape = factory.testDataShape
    total_time = 0
    for img, img_name in images:
        print(f"Processing {img_name}")
        result = np.zeros((img.shape[0], 4))
        FUNS = np.zeros(((img.shape[0]), 21))
        start = time.time()
        for i in range(img.shape[0]):
            targetPixel = img[i]
            res = lsq_linear(designMatrix, targetPixel, bounds = (0, np.inf), lsmr_tol = 'auto', verbose = 0)
            result[i] = res.x
            FUNS[i] = res.fun
        end = time.time()
        img_name = os.path.basename(img_name)
        np.save("sciOptimizeRes/" + img_name + ".npy", result) 
        avgFuns = np.average(FUNS, axis = 1).reshape(image_shape)
        normalizedError = np.absolute(avgFuns) / np.amax(avgFuns) * 255
        np.save("sciOptimizeRes/" + img_name + "_FUNS.npy", normalizedError)
        print(f"Time to find output of each pixel {end - start}")
        total_time += end-start
    print(f"Finised generating all images: Average creation time: {total_time/5}")



def doUnmixing(data, designMatrix):
    img = data.getImage()
    img_name = data.testDataName
    imageShape = data.testDataShape
    
    plt.plot(designMatrix[:,2], label = "BSA", color = "blue")
    plt.plot(designMatrix[:,0], label = "DNA", color = "red")
    plt.plot(designMatrix[:,1], label = "OA", color = "green")
    plt.plot(designMatrix[:,3], label = "Background", color = "black")
    plt.legend()
    plt.show()

    result = np.zeros((img.shape[0],4))
    FUNS = np.zeros(((img.shape[0]), 21))
    full_start = time.time
    for i in range(img.shape[0]):
        targetPixel = img[i]
        res = lsq_linear(designMatrix, targetPixel, bounds = (0, np.inf), lsmr_tol = 'auto', verbose = 0)
        result[i] = res.x
        FUNS[i] = res.fun
    full_end = time.time
    np.save("sciOptimizeRes/" + img_name + ".npy", result) 
    avgFuns = np.average(FUNS, axis = 1).reshape(imageShape)
    normalizedError = np.absolute(avgFuns) / np.amax(avgFuns) * 255
    np.save("sciOptimizeRes/" + img_name + "_FUNS.npy", normalizedError)
    print(f"Time to find output of each pixel {full_end - full_start}")

if __name__ == "__main__":
    main()