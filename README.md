The final project for my DATA3402 class; an application of all required skills. Aims to sufficiently complete the closed Kaggle Challenge "Forest Fire Prediction."
https://www.kaggle.com/competitions/forest-fire-prediction/overview
# Forest Fire Prediction
**Tasks**
Use the various parameters such as relative humidity, temperature, wind speed, etc. to develop a program to predict whether or not there will be a forest fire.

## Summary
***************************************************
**Problem Formulation**
Define: Correctly perform binary classificaton
Input: CSV file of features, output: class in last column

**Data**
Type: CSV
Size and instances: Train CSV with 200 examples, Test CSV with 50 examples. Train was split 80/20 for validation.
Data is availible for download in the repository, as well as through the link at the top of the README file.

**Data Visualization**
Show a few visualization of the data and say a few words about what you see.

**Preprocessing / Clean up**
Dataset was very clean; just needed to do some data conversions, encoding of categorical variables, and feature selection/reduction.

**Models**
I tried Random Forest and Support Vector Machine. Through trials, the SVM seemed less prone to overfit and favorable towards the scaled data, so I stuck with that one.
SVC(C=1.7, gamma='auto')

**Training**
GridSearchCV
Software/Hardware: scikit-learn on WSL through Ubuntu and Jupyter Notebook.
I ran into overfitting problems and spent a while adjusting the data as well as the hyperparameters. I stopped once I found a good balance of f1, accuracy, and no overfitting.
I ran into some problems with reproduciblity, and had to adjust to new parameters.

**Performance**
Clearly define the key performance metric(s).
Show/compare results in one table.
Show one (or few) visualization(s) of results, for example ROC curves.

**Conclusions**
SVM is a viable option for forest fire prediction, but more work needs to be done in order to improve it without overfitting. 
Future Work:
Trying a simple deep learning model may do the trick.

**How to reproduce results**
Remove all categorical and calculated features, scale the data, then do a GridSearchCV with an SVM.
This dataset is very lightweight, you sshouldn't need any special hardware. 




**Overview of files in repository**
train.csv: the training data.
test.csv: the testing data.
fire_test_preds.csv: the predictions for the test data in the proper submission form.
FireKaggle.ipynb: my comprehensive code for the project.

**Software Setup**
import pandas as pd
from zipfile import ZipFile
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix,  accuracy_score
from sklearn.ensemble import RandomForestClassifier
