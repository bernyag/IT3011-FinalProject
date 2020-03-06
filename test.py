import imageio
import numpy as np
import os
from skimage import data, color, exposure
from sklearn.model_selection import  train_test_split

import matplotlib.pyplot as plt

def get_all_data(directory=f'{os.getcwd()}\\images_no_copies'):
    labels = {}
    for entry in os.scandir(directory):
        label = entry.name
        images = []

        for picture in os.scandir(entry):
            image = imageio.imread(picture.path)
            training_digit = color.rgb2gray(image)
            images.append(training_digit)
        
        labels[label] = images
        print(labels)
    
    return labels