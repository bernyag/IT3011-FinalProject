import imageio
import numpy as np
import os
from skimage import data, color, exposure
from sklearn.model_selection import  train_test_split

import matplotlib.pyplot as plt

def get_all_data(directory=f'{os.getcwd()}/images_no_copies'):
    labels = {} 
    for entry in os.scandir(directory):
        label = entry.name      # The operand / operator. Example: '2'
        images = []             

        for picture in os.scandir(entry):
            image = imageio.imread(picture.path)
            training_digit = color.rgb2gray(image)
            images.append(np.insert(training_digit,45,255, axis=1)) #append them with a white column to the right to prevent numbers joining
        labels[label] = images    
    return labels
def getList(dict): 
    return dict.keys() 

#find . -name "*.DS_Store" -type f -delete  

