# Simple CNN for the MNIST Dataset
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import numpy as np
import csv
from matplotlib import pyplot as plt
import cv2

values=['div','times','+','-','0','1','2','3','4','5','6','7','8','9']

# load dataset
file = np.genfromtxt('./equations/good.csv', delimiter=',')
# file = csv.reader("good.csv", delimiter=',')
data = file[1:, 1:]
changedData=[]
for img in data:
	img=img.reshape(45,46)
	img=np.resize(img,(45,45))
	img=img.flatten()/255
	changedData.append(img)
changedData=np.asarray(changedData)

labellist = []
with open('./equations/good.csv') as f:
    reader = csv.reader(f, delimiter=",")
    for i in reader:
        labellist.append(list(i)[0])
labels = np.asarray(labellist)[1:]
newLabels=[]
for each in labels:
	currLabel=values.index(each)
	newLabels.append(currLabel)
labels=np.asarray(newLabels)

x_train, x_test, y_train, y_test = train_test_split(changedData, labels, test_size = 0.2, random_state = 101)
x_train = x_train.reshape(x_train.shape[0], 45, 45,1)
x_test = x_test.reshape(x_test.shape[0], 45, 45,1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# normalize inputs from 0-255 to 0-1
x_train = x_train / 255
x_test = x_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
print(num_classes)

model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(45, 45, 1), activation='relu'))
model.add(MaxPooling2D())


model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# Fit the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=15, batch_size=40)
# Final evaluation of the model
scores = model.evaluate(x_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))

model.save('model.h5')
print("Model saved as model.h5")
