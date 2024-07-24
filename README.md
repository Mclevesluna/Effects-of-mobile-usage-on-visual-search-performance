# Project Overview
1. Project Overview
Our project consists of two tasks, each conducted in a separate JupyterLab Notebook (ClassAssignmentPart1.ipynb and ClassAssignmentPart2.ipynb).

For the first task, our goal was to study whether users with high mobile usage find a given visual search target faster than their counterparts. We also aimed to assess if there is any significant interaction with other independent variables (e.g., Gender, Age, etc.). To achieve this, we followed a series of steps, including checking data quality, analyzing data relationships/distributions, exploring correlations and associations (regressions), and running inferential statistical tests, such as ANOVA.

For the second task, we aimed to analyze the types of variables and data within four datasets, identifying relationships, patterns, and potential classification problems. To accomplish this, we followed a series of steps to determine data/variable types, identify linearity and patterns, and explore classification problems. For specific datasets, we also explored models such as decision trees, multiple regressions, and KNN predictors.

2. Installation Instructions
Each Jupyter Notebook uses a series of libraries. Ensure the following libraries are installed
```
pip install -r requirments.txt
```
if you nrrf gpu, OS
ClassAssignmentPart1.ipynb:

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
import statsmodels.api as sm

import statsmodels.api as sm 
from statsmodels.formula.api import ols 
from scipy import stats
from scipy.stats import bartlett
from scipy.stats import kruskal
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

ClassAssignmentPart2.ipynb:

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.api as sm 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers.legacy import SGD
from tensorflow.keras.utils import to_categorical
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text, plot_tree
from sklearn.metrics import accuracy_score, mean_squared_error
from mlxtend.plotting import plot_decision_regions
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


3. Data sets used

The first uses one data set and the second task uses four data sets. For the first task, we have also created an adjusted version of the data set used that includes all additional columns and variables created during the analysis process. 

These are the relevant data sets for each notebook:

ClassAssignmentPart1.ipynb:MASTER_PhonesmartdataAll_CCI_AdvStats.csv, df_finalAdjusted.csv

ClassAssignmentPart2.ipynb:MASTER_PhonesmartdataAll_CCI_AdvStats.csv, data.csv, Housing.csv, wine_data.csv

All data sets were supplied by the Creative Computing Institute at University of the Art London (2023).

4. Structure and usage guide for notebooks
The notebooks have very detailed comments that justify all tests/models that were applied/created. They also have detailed conclusions that the user can read through for each section.

This is the structure of sections for each notebook:

ClassAssignmentPart1.ipynb:

    0. Importing Libraries
    1. Check Data Quality
    2. Data Relationship/Distribution
    3. Correlation Check: Produce a bivariate correlation table between Age, STAI, BRIEF_Total, DailyAvgMins and VS_RT_correct_Single.
    4. Linear Regression: Perform a linear regression to see if DailyAvgMins predicts VS_RT_correct_Single
    5. Multiple Regression: Add predictors Age, GenderNum, STAI, BRIEF_Total and DailyAvgPickups to the multiple regression model. Does the amount of variance accounted for in the outcome increase? Is DailyAvgMins a significant predictor of the outcome?
    6. Scenario 1 for inferential statistics
    7. Scenario 2 for inferential statistics
    8. Print out final adjusted MASTER data set

(For this first task, all sources used are recorded within each individual section)

ClassAssignmentPart1.ipynb:

    0. Import Libraries and structure of this notebook
    1. Reading the Data
    2. Finding Missing Values
    3. Determining types of data and variables
    4. Data set 1: DATA (MUSIC) - ANALYSIS 
    5. Data set 2: HOUSING - ANALYSIS 
    6. Data set 3: WINE - ANALYSIS
    7. Data set 4: MASTER (CELLPHONE) - ANALYSIS
    8. Sources
       
To look through conclusions, challenges and all sources used please refer to each individual notebook where you will find detailed comments for each section mentioned in the above structures.
