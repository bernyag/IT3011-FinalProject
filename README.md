# IT3011-GroupProject

## Project description


### Prerequisites
We use anaconda to manage all dependencies and packages.
Make sure to have installed Anaconda 4.8.3 or later.

opencv <br>
scikit-learn >= 0.22
```
conda install opencv
conda install scikit-learn=0.22
```






## Files
### equations/calculate.py
Once all of the elements of the equation have been identified, the labels are converted into a string, for example “0/7”. This string is then passed into the solve() function. The calculator can take an equation of any length. Because of this we used a stack and a queue to convert the equation into postfix form, which would also take into account the precedence of / and * over + and -. Once the digits had been converted to integers and the whole equation was in postfix form, the entire equation could be solved using basic math and the solution was printed.

### equations/functions.py
This notebook has three main methods: 
get_array_from_images: this is used to get the images from our dataset and convert them to arrays into a saved dictionary     containing all of the images. This method was used in many of our notebooks so that we had the data in the format we           neded. 
split_data: this method splits part of the data into test and train, we used this in the other notebooks to train our         model.
    
    
### equations/kfold_test.ipynb
This notebook was used to test cross validation and some metrics for our models. It was not used for the solution.

### equations/imageReading.ipynb
This was a notebook used to understand our dataset. Here we did some experiments on how to read it and how the values looked. It was not used for the solution.

### equations/my_notebook.ipynb
Description here

### equations/test_run.py
Description here

### cNNEvaluation.ipynb
In order to run cNNEvaluation.ipynb you must have the folder images_no_copies. This file uses keras to train a convolutional neural network based on the images of handwritten digits and operations. Once the data has been read in and formatted, the training of the model will begin and you will be able to see the accuracy after each epoch. For best results, use epoch=15, batchsize=40.This file also provides additional performance scores which can be run to test the accuracy, precision, f1 score and recall of the model. The confusion matrix can also be produced in this file.

### localize2.ipynb
Description here

### my_notebook.ipynb
Description here

### run.ipynb
Description here

### runFullEquation.py
In order to run runFullEquation.py you must have the foler generated_images which was too large to upload the entire folder to GitHub. To run runFullEquation.py the command is "python3 runFullEquation.py" . This file uses keras to train a convolutional neural network based on the images of entire equations of handwritten digits and operations. Once the data has been read in and formatted, the training of the model will begin and you will be able to see the accuracy after each epoch. 

### testcNN.py
To run testcNN.py, traincNN.py must be run first to create the model.h5 file. This file allows the user to test the predict capability of the model, without the need to retrain each time. To test different values simply change the directory name.

### traincNN.py
In order to run trainCNN.py you must have the large good.csv file which was too large to upload to GitHub. If this is not the case, please run cNNEvaluation.ipynb instead. To run traincNN.py the command is "python3 traincNN.py" . This file uses keras to train a convolutional neural network based on the images of handwritten digits and operations. Once the data has been read in and formatted, the training of the model will begin and you will be able to see the accuracy after each epoch. For best results, use epoch=15, batchsize=40.



