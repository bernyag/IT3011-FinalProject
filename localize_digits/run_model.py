import cv2
from localizer_functions import parse_equation
import pickle
path='../generated_images/0+8/9.png'
image=cv2.imread(path) 
symbols=parse_equation(image)
#print(symbols)

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
classifier= pickle.load(open('model.sav', 'rb'))
result=classifier.predict(testDigits)
result=str(result)
print(result)

