import cv2
import numpy as np
import keras
import matplotlib.pyplot as plt

classifier = keras.models.load_model('../model.h5') # edit this if your model is in a different directory
values=['div','times','+','-','0','1','2','3','4','5','6','7','8','9']


from localizer_functions import parse_equation

path='../generated_images/0+5/1.png'
image = cv2.imread(path) 
symbols=parse_equation(image)


testDigits = []
for picture in symbols:
    img_data = []
    resized_img=(np.resize(picture, (45,45)))
    img_data.append(resized_img.flatten() / 255) # flatten to 784 and normalize values
    img_data = np.asarray(img_data)
    #img_data=np.resize(img_data,(2025))
    img_data = img_data.reshape(1, 45, 45, 1)
    img_data = img_data.astype('float32')
    result=classifier.predict(img_data)
    final=values[np.argmax(result)]

    #picture = cv2.resize(picture, dsize=(2025), interpolation=cv2.INTER_CUBIC)
    testDigits.append(final)
    
#for each in symbols:

# load the model from disk

print(testDigits)

# Read in an image
#img = cv2.imread('./localize_digits/1.png', cv2.IMREAD_GRAYSCALE)

# Preprocessing
#img_data = []
#resized_img = cv2.resize(img, (45,45))
#img_data.append(resized_img.flatten() / 255) # flatten to 784 and normalize values
#img_data = np.asarray(img_data)
#img_data = img_data.reshape(img_data.shape[0], 45, 45, 1)
#img_data = img_data.astype('float32')

#result = model.predict(img_data)
