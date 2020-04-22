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
Description here

### equations/kfold_test.ipynb
Description here

### equations/imageReading.ipynb
Description here

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



