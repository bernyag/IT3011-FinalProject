# get data
from functions import get_all_data, split_data, reshape
import cv2
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn import datasets, svm, metrics
import os
import cv2
import pickle

data = get_all_data()
dev, test = split_data(data, 0.2)
training_data, training_labels = reshape(dev)
test_data, test_labels = reshape(test)

classifier = svm.SVC(gamma=0.001)
classifier.fit(training_data, training_labels)
# save the model to disk
filename = 'model.sav'
pickle.dump(classifier, open(filename, 'wb'))
 




#score = classifier.score(test_data, test_labels)

#print(score)