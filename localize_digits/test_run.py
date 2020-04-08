# get data
from functions import get_all_data, split_data, reshape
import cv2
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn import datasets, svm, metrics
import os
import cv2
from localizer_functions import parse_equation

path='../generated_images/0+8/0.png'
image=cv2.imread(path) 
symbols=parse_equation(image)
print(symbols)

##########

data = get_all_data()
dev, test = split_data(data, 0.2)
training_data, training_labels = reshape(dev)
test_data, test_labels = reshape(test)

classifier = svm.SVC(gamma=0.001)
classifier.fit(training_data, training_labels)


testDigits = []
for picture in symbols:
    #testDigit = cv2.imread(picture,cv2.IMREAD_GRAYSCALE) /255
    #testDigit=testDigit.transpose(2,0,1).reshape(-1,testDigit.shape[1])
    picture=picture/255
    picture=picture.flatten()
    picture.resize(2025)
    testDigits.append(picture)
print("here")
#for each in symbols:
result=classifier.predict(testDigits)
print(result)

#score = classifier.score(test_data, test_labels)

#print(score)