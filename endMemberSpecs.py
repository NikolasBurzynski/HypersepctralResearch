from pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import cv2 as cv

'''
endMemberSpecs.py

This module includes code that is responsible for the creation of the "design matrix." This code uses the Pipeline object. See Pipeline.py for reference
  
'''

def norm_avg(flattened):
    flattened[flattened < 0] = 0
    avg = np.average(flattened, axis = 0)
    print(avg.shape)
    min = np.amin(avg)
    # min = 0
    avg = avg - min # We probably need to keep these.
    max = np.amax(avg)
    normed = avg / max
    return normed 


def threshold(flat_data, flat_mask):
    flattened = np.delete(flat_data, np.where(flat_mask == 0), axis = 0)
    return flattened

def generate_spec(data, do_thresh=False, thresh=None):
    flat_data = np.reshape(data, (-1, data.shape[-1]))[:, :17]
    if do_thresh:
        flat_thresh = np.reshape(thresh, (-1, thresh.shape[-1]))
        return norm_avg(threshold(flat_data, flat_thresh))
    return norm_avg(flat_data)

def get_background():
    STACKS = Pipeline('Data_Files/Endmembers/Stacks/', doEdit=True, changeVals=False).all_imgs
    BACKGROUNDS = [io.imread(f'Data_Files/Endmembers/Backgrounds/background_{idx+1}.jpg') for idx in range(len(STACKS))]

    print('==========')
    print(len(STACKS))
    print(len(BACKGROUNDS))
    print('==========')
    assert len(STACKS) == len(BACKGROUNDS)

    parts = []
    for stack, background_layer in zip(STACKS, BACKGROUNDS):
        _, thrsh = cv.threshold(background_layer, 0, 1, cv.THRESH_BINARY + cv.THRESH_OTSU)
        plt.imshow(thrsh)
        plt.show(block=True)
        stack = np.reshape(stack, (-1, stack.shape[-1]))
        thrsh = np.reshape(thrsh, (-1))
        background_pixels = stack[thrsh == 0]

        endmember_curve = norm_avg(background_pixels)
        parts.append(endmember_curve)
    parts = np.array(parts)
    print('==========')
    print(parts.shape)
    avg = np.mean(parts, axis=0)
    print(avg.shape)
    print('==========')

    return avg



def main():
    print("Generating Pure Endmember Spectra")
    pipe = Pipeline("Data_Files/Endmembers/", doEdit=True, changeVals=False)
    print(pipe.filenames)
    BSA = pipe.all_imgs[0].astype(np.float32)
    DNA = pipe.all_imgs[1].astype(np.float32)
    OA = pipe.all_imgs[2].astype(np.float32)
    WATER = pipe.all_imgs[5].astype(np.float32)
    BSA_THRESH = io.imread('Data_Files/Endmembers/Thresholds/BSA-1-THRSD.jpg')
    DNA_THRESH = io.imread('Data_Files/Endmembers/Thresholds/DNA-1-THRSD.jpg')

    BACKGROUND_SPEC = get_background()[0:17]


    BSA_SPEC = generate_spec(BSA, True, BSA_THRESH)[0:17]
    DNA_SPEC = generate_spec(DNA, True, DNA_THRESH)[0:17]
    OA_SPEC = generate_spec(OA)[0:17]
    WATER_SPEC = generate_spec(WATER)[0:17]
    WATER_SPEC = WATER_SPEC / np.amax(WATER_SPEC)
    print(WATER_SPEC.shape)
    print(BACKGROUND_SPEC.shape)
    BACK_WATER_AVG_SPEC = (BACKGROUND_SPEC + WATER_SPEC) / 2


    plt.plot(BSA_SPEC, label = 'BSA', color = "blue")
    plt.plot(OA_SPEC, label = 'OA', color = "green")
    plt.plot(DNA_SPEC, label = 'DNA', color = "red")
    plt.plot(BACK_WATER_AVG_SPEC, label = 'BACKGROUND', color = "black")

    plt.legend()
    plt.show()
    specs = [DNA_SPEC, OA_SPEC, BSA_SPEC, BACK_WATER_AVG_SPEC]
    print("specs", np.array(specs, dtype=np.float64))
    npSpecs = np.array(specs, dtype=np.float64).T
    print("npSpecs", npSpecs)
    print(npSpecs.shape)
    np.save("Data_Files/17-withDNA-DesignMatrix_All_1_Norm_background_plus_water.npy", npSpecs)



if __name__ == "__main__":
    main()