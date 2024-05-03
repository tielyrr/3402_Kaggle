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

***********************************************************************
**Data Visualization**
Quite a bit of the data is skewed, robust scaling would be best, if any.
![prelim](https://github.com/tielyrr/3402_Kaggle/assets/143365566/5e2d11be-40a1-4775-afa6-40aee6ede5b9)
Many of the features are quite distinct.
![comps](https://github.com/tielyrr/3402_Kaggle/assets/143365566/8eb92471-685c-4fb6-96c6-ae8247cdc77e)



**Preprocessing / Clean up**
Dataset was very clean; just needed to do some data conversions, encoding of categorical variables, and feature selection/reduction. I had initially kept all features besides the day and year, but that later, to reduce risk of overfitting, I reduced it drastically, undoing quite a bit of my original work.


**Models**
I tried Random Forest and Support Vector Machine. Through trials, the SVM seemed less prone to overfit and favorable towards the scaled data, so I stuck with that one.
SVC(C=1.7, gamma='auto')

Then I ran it again with even less features and got a great result.

SVC(C=0.5, gamma='auto', kernel='linear')



**Training**
GridSearchCV

Software/Hardware: scikit-learn on WSL through Ubuntu and Jupyter Notebook.

I thought I ran into overfitting problems and spent a while adjusting the data as well as the hyperparameters. I stopped once I found a good balance of f1, accuracy.
I ran into some problems with reproduciblity, and had to adjust to new parameters multiple times.

When I ran it the last time with minimal features, there was no overfitting and perfect accuracy. 


## Performances

**First run after feature reducing and tuning:**

![Screenshot 2024-05-02 230833](https://github.com/tielyrr/3402_Kaggle/assets/143365566/83a46f47-2369-4575-9aec-1ff106489cd5)
![roc](https://github.com/tielyrr/3402_Kaggle/assets/143365566/3d7958a5-73b4-432b-9c94-db4ecb03c53a)
![Screenshot 2024-05-03 103648](https://github.com/tielyrr/3402_Kaggle/assets/143365566/ba07d361-4bf0-4215-9566-26f0f85d0525)

**Second run with only the DC, FFMC, and ISI features with tuning:**

![Screenshot 2024-05-03 171836](https://github.com/tielyrr/3402_Kaggle/assets/143365566/f9a085df-fbce-4597-bdcb-2225321e20d2)
![roc2](https://github.com/tielyrr/3402_Kaggle/assets/143365566/d352ed3d-05a4-4c5d-b9b5-4f80860aeb58)


********************************************
**Kaggle Challenge Results**
![Screenshot 2024-05-03 175445](https://github.com/tielyrr/3402_Kaggle/assets/143365566/32f5f1b6-d8fa-49b6-9e55-098ece094f8d)


***********************************************
**Conclusions**

SVM is a viable option for forest fire prediction. 

In the future, trying a simple deep learning model may work with a larger and more variable dataset.


**How to reproduce results**
To reproduce the final model's results;

- Remove all categorical variables, keep the DC, FFMC, and ISI and Classes. 
- Do not scale the data
- GridSearchCV with an SVM.
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

