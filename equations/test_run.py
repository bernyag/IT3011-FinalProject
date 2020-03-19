# get data
import functions as f
#import cv2
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn import datasets, svm, metrics

data = f.get_array_from_images('images_no_copies')
dev, test = f.split_data(data, 0.2)
training_data, training_labels = f.reshape(dev)
test_data, test_labels = f.reshape(test)

classifier = svm.SVC(gamma=0.001)
classifier.fit(training_data, training_labels)
score = classifier.score(test_data, test_labels)

print(score)