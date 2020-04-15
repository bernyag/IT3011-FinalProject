import imageio
import numpy as np
import os
from skimage import data, color, exposure
from sklearn.model_selection import  train_test_split
import matplotlib.pyplot as plt
import cv2


def get_cv_array_from_images(fName):
    directory=f'{os.getcwd()}/'+fName
    labels = {} 
    for entry in os.scandir(directory):
        label = entry.name      # The operand / operator. Example: '2'
        images = []             

        for picture in os.scandir(entry):
            image = cv2.imread(picture.path)
            images.append(image) #append them with a white column to the right to prevent numbers joining
        labels[label] = images  
    print('Test')  
    return labels

'''
def get_all_data(fName):
    directory=f'{os.getcwd()}/'+fName
    labels = {} 
    for entry in os.scandir(directory):
        label = entry.name      # The operand / operator. Example: '2'
        
        images = []             
        for picture in os.scandir(entry):
            training_digit = cv2.imread(picture.path, cv2.IMREAD_GRAYSCALE) / 255
            images.append(training_digit)
        
        labels[label] = images
    return labels
'''

def split_data(data:dict, test_ratio:float = 0.2) -> (dict, dict):
    dev = {}
    test = {}

    for label, pictures in data.items():
        test_size = int(len(pictures) * test_ratio)
        test[label] = pictures[:test_size]
        dev[label] = pictures[test_size:]
    
    return dev, test


def getList(dict): 
    return dict.keys()

def reshape(data:dict) -> (np.array, np.array):
    length = sum(len(lst) for lst in data.values())

    # labels = np.empty(length, dtype='<U1')
    # y = np.ndarray(shape =(45, length), dtype='int32')
    y = [None] * length
    labels = [None] * length

    i = 0
    for label, pictures in data.items():
        for picture in pictures:
            labels[i] = label
            y[i] = picture.flatten()
            i+=1

    return np.array(y), np.array(labels)

#find . -name "*.DS_Store" -type f -delete  

