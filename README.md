# IT3011-GroupProject

## Project description

Final proyect for Machine Learning and Applications, where a Python application was created to scan handwritten numbers using Machine Learning to evaluate simple math  equations. 

### Presentation
https://docs.google.com/presentation/d/1kcP9wKIxpnOCwK6cCWkC5i9YpCF5t1ZneJ36vzg2_5Y/edit#slide=id.g73a25662d7_4_34

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
Initial notebook to test quick implementations of KNN and later SVM, with sklearn.

### equations/test_run.py
Similiar to equations/my_notebook.ipynb, the script runs an implemenation of SVM and prints out its metrics.

### cNNEvaluation.ipynb
In order to run cNNEvaluation.ipynb you must have the folder images_no_copies. This file uses keras to train a convolutional neural network based on the images of handwritten digits and operations. Once the data has been read in and formatted, the training of the model will begin and you will be able to see the accuracy after each epoch. For best results, use epoch=15, batchsize=40.This file also provides additional performance scores which can be run to test the accuracy, precision, f1 score and recall of the model. The confusion matrix can also be produced in this file.

### localizer
#### localizer/localizer_functions.py
This file has a collection of all the functions needed for the digit localization. This requires openCV (`cv2`) to be installed. The main function in this file is called `parse_equation`, the other functions are mainly helper functions. This function splits an openCV image in a list of the individual symbols our algorithm has detected.
Additionally, this file offers `get_all_data_cv` which reads all images from a folder recursively. This works similar to the other `get_all_data`, however it uses the openCV image format.

#### localizer/localize_tester.py
Python file to test the localize functions and used for debugging. It saves misclassified (= not correct number of symbols) equations parts into an err/ folder.

#### localizer/nb_localize.ipynb
Jupyter notebook in which the `localizer_functions` are used interactively. Also used for debugging and creating the `detected_*` images. An image can be read in the first line and then subsequently analyzed. Additionally, similar to `localize_tester.py`, all images from a folder can be splitted.

#### localizer/first_tests_localizer.ipynb
The first tests in getting used to openCV and how we can use it to split images.

### my_notebook.ipynb
copy of equations/my_notebook.ipynb, but with a focus only on SVM.

### run.ipynb
Notebook version of trainCNN.py, to highlight every step of the training.

### runFullEquation.py
In order to run runFullEquation.py you must have the foler generated_images which was too large to upload the entire folder to GitHub. To run runFullEquation.py the command is "python3 runFullEquation.py" . This file uses keras to train a convolutional neural network based on the images of entire equations of handwritten digits and operations. Once the data has been read in and formatted, the training of the model will begin and you will be able to see the accuracy after each epoch. 

### testcNN.py
To run testcNN.py, traincNN.py must be run first to create the model.h5 file. This file allows the user to test the predict capability of the model, without the need to retrain each time. To test different values simply change the directory name.

### traincNN.py
In order to run trainCNN.py you must have the large good.csv file which was too large to upload to GitHub. If this is not the case, please run cNNEvaluation.ipynb instead. To run traincNN.py the command is "python3 traincNN.py" . This file uses keras to train a convolutional neural network based on the images of handwritten digits and operations. Once the data has been read in and formatted, the training of the model will begin and you will be able to see the accuracy after each epoch. For best results, use epoch=15, batchsize=40.



