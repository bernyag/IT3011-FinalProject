# get data
from functions import get_all_data, split_data, reshape
import cv2
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn import datasets, svm, metrics
import os
import cv2
from localize2 import run

data = get_all_data()
dev, test = split_data(data, 0.2)
training_data, training_labels = reshape(dev)
test_data, test_labels = reshape(test)

classifier = svm.SVC(gamma=0.001)
classifier.fit(training_data, training_labels)

symbols = run(f'{os.getcwd()}/generated_images/0+1')

testDigits = []
for picture in symbols:
    testDigit = cv2.imread(picture.path,cv2.IMREAD_GRAYSCALE) /255
    #testDigit=testDigit.transpose(2,0,1).reshape(-1,testDigit.shape[1])
    testDigit=testDigit.flatten()
    testDigits.append(testDigit)
result=classifier.predict(testDigits)
print(result)

#score = classifier.score(test_data, test_labels)

#print(score)