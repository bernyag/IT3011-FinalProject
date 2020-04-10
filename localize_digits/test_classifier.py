import cv2
from localizer_functions import parse_equation
import pickle
from matplotlib import pyplot as plt
import os
import functools
import collections

base='../images_no_copies/'
symbol = '+'
path = f"{base}{symbol}"
symbols = []
i = 0

num_test = 2000
for entry in os.scandir(path):  
    i += 1  
    symbols.append(cv2.imread(entry.path))
    if i == num_test:
        break
#rint(image)
#symbols=[image]


##########


testDigits = []
for picture in symbols:
    #testDigit = cv2.imread(picture,cv2.IMREAD_GRAYSCALE) /255
    #testDigit=testDigit.transpose(2,0,1).reshape(-1,testDigit.shape[1])
    picture=picture/255
    #picture=picture.flatten()
    picture.resize(2025)

    #picture = cv2.resize(picture, dsize=(2025), interpolation=cv2.INTER_CUBIC)
    testDigits.append(picture)
    
#for each in symbols:

# load the model from disk
classifier= pickle.load(open('../model.sav', 'rb'))
result=classifier.predict(testDigits)

print(result)

print(collections.Counter(result))
