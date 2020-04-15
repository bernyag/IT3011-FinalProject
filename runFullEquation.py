# Simple CNN for the MNIST Dataset
import keras
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import numpy as np
import csv
from matplotlib import pyplot as plt
import cv2
import os

values=[]

train_data = []
labels = []

train_folder='./generated_images/'

print("Creating input data...")
for foldername in os.listdir(train_folder):
    values.append(foldername)
    for filename in os.listdir(train_folder + foldername):
        img = cv2.imread(train_folder + foldername + "/" + filename, cv2.IMREAD_GRAYSCALE)
        currLabel=values.index(foldername)
        resized_img = cv2.resize(img, (45,135))
        img_data = resized_img.flatten() / 255 # flatten to 784 and normalize values
        train_data.append(img_data)
        labels.append(currLabel)

train_data = np.asarray(train_data)
labels = np.asarray(labels)

print("Created input data with shape: %s" % (train_data.shape,))
print("Created label data with shape: %s" % (labels.shape,))

x_train, x_test, y_train, y_test = train_test_split(train_data, labels, test_size = 0.2, random_state = 101)
x_train = x_train.reshape(x_train.shape[0], 45,135 ,1)
x_test = x_test.reshape(x_test.shape[0], 45, 135,1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# normalize inputs from 0-255 to 0-1
x_train = x_train / 255
x_test = x_test / 255
initialsplit=x_train
# one hot encode outputs
y_train = np_utils.to_categorical(y_train,len(labels))
y_test = np_utils.to_categorical(y_test,len(labels))
num_classes = y_test.shape[1]


model = keras.models.Sequential()
model.add(keras.layers.Conv2D(64, kernel_size=(3,3), activation = 'relu', input_shape=(45, 135 , 1) ))
model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))

model.add(keras.layers.Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))

model.add(keras.layers.Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation = 'relu'))
model.add(keras.layers.Dropout(0.20))

model.add(keras.layers.Dense(len(labels), activation = 'softmax'))
model.compile(loss = keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])


# Fit the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=50)
# Final evaluation of the model
scores = model.evaluate(x_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))

Y_pred = model.predict(x_test)
y_pred = np.argmax(Y_pred, axis=1)

model.save('model.h5')
print("Model saved as model.h5")