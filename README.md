The final project for my DATA3402 class. Aims to sufficiently complete the closed Kaggle Challenge "Forest Fire Prediction."
https://www.kaggle.com/competitions/forest-fire-prediction/overview
# Forest Fire Prediction
**Tasks**
Use the various parameters such as relative humidity, temperature, wind speed, etc. to develop a program to predict whether or not there will be a forest fire.

## Summary
***************************************************
**Problem Formulation**
Correctly perform binary classificaton

Input: CSV file of features

Output: class in last column


**Data**
Type: CSV

Size and instances: Train CSV with 200 examples, Test CSV with 50 examples. Train was split 80/20 for validation.

Data is availible for download in the repository, as well as through the link at the top of the README file.

*******************************************************************************************************
The kaggle site does not tell me what the acronyms stand for. From a brief search, Fire Weather Indices (FWI) seem to be uniform among cities, states, and countries. This was the first guide to the acronym meanings I found:
https://cwfis.cfs.nrcan.gc.ca/background/summary/fwi

*Fine Fuel Moisture Code (FFMC)*
-a numeric rating of the moisture content of litter and other cured fine fuels. This code is an indicator of the relative ease of ignition and the flammability of fine fuel.

*Duff Moisture Code(DMC)*
-a numeric rating of the average moisture content of loosely compacted organic layers of moderate depth. This code gives an indication of fuel consumption in moderate duff layers and medium-size woody material.


*Drought Code(DC)*
-a numeric rating of the average moisture content of deep, compact organic layers. This code is a useful indicator of seasonal drought effects on forest fuels and the amount of smoldering in deep duff layers and large logs.


*Initial Spread Index(ISI)*
-a numeric rating of the expected rate of fire spread. It is based on wind speed and FFMC. Like the rest of the FWI system components, ISI does not take fuel type into account. Actual spread rates vary between fuel types at the same ISI.


*Buildup Index(BUI)*
-a numeric rating of the total amount of fuel available for combustion. It is based on the DMC and the DC. The BUI is generally less than twice the DMC value, and moisture in the DMC layer is expected to help prevent burning in material deeper down in the available fuel.


*Fire Weather Index(FWI)*
-a numeric rating of fire intensity. It is based on the ISI and the BUI, and is used as a general index of fire danger throughout the forested areas of Canada.



I can also only assume that for the Classes, 1.0 means there was a fire and 0.0 means there wasn't


**Data Visualization**
![prelim](https://github.com/tielyrr/3402_Kaggle/assets/143365566/5e2d11be-40a1-4775-afa6-40aee6ede5b9)
Quite a bit of the data is skewed, robust scaling would be best.
![cat](https://github.com/tielyrr/3402_Kaggle/assets/143365566/83ab2dd1-49d7-496d-a282-f2cc69b4c5e1)


**Preprocessing / Clean up**
Dataset was very clean; just needed to do some data conversions, encoding of categorical variables, and feature selection/reduction. I had initially kept all features besides the day and year, but that later led to overfit so I reduced it drastically, undoing quite a bit of my original work.


**Models**
I tried Random Forest and Support Vector Machine. Through trials, the SVM seemed less prone to overfit and favorable towards the scaled data, so I stuck with that one.
SVC(C=1.7, gamma='auto')


**Training**
GridSearchCV

Software/Hardware: scikit-learn on WSL through Ubuntu and Jupyter Notebook.

I ran into overfitting problems and spent a while adjusting the data as well as the hyperparameters. I stopped once I found a good balance of f1, accuracy, and no overfitting.
I ran into some problems with reproduciblity, and had to adjust to new parameters.


**Performance**


![Screenshot 2024-05-02 230833](https://github.com/tielyrr/3402_Kaggle/assets/143365566/83a46f47-2369-4575-9aec-1ff106489cd5)
![roc](https://github.com/tielyrr/3402_Kaggle/assets/143365566/3d7958a5-73b4-432b-9c94-db4ecb03c53a)

![Screenshot 2024-05-03 103648](https://github.com/tielyrr/3402_Kaggle/assets/143365566/ba07d361-4bf0-4215-9566-26f0f85d0525)


**Conclusions**
SVM is a viable option for forest fire prediction, but more work needs to be done in order to improve accuracy without overfitting. 

Future Work:

Trying a simple deep learning model may do the trick.


**How to reproduce results**
Remove all categorical and calculated features, scale the data, then do a GridSearchCV with an SVM.
This dataset is very lightweight, you shouldn't need any special hardware. 



******************************************************************************************
**Overview of files in repository**
train.csv: the training data

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

