# Data Science Project: Mobile Usage and Visual Search Performance

## Overview
This project aims to study if users with high mobile usage find a given visual search target faster than their counterparts and to assess if there is any significant interaction with variables like gender and age. To achieve this, we followed a series of steps, including checking data quality, analyzing data relationships/distributions, exploring correlations and associations (regressions), and running inferential statistical tests, such as ANOVA.

## Dataset
Data set for this project is in folder titled "data", corresponds to a study conducted by xxxx and was provided by Creative Computing Institute at University of the Art London (2023). 
We have also created an adjusted version of the data set used that includes all additional columns and variables created during the analysis process.

## Preferred Language
Python

## Project Tasks

### 1. Data Quality Check
- **Objective**: Create a Data Pre-processing pipeline to ensure the dataset is ready for analysis.
- **Steps**:
  - Clean and preprocess the data.
  - Confirm the data quality and record the data shape.
  - Includes necessary comments and justifications throughout the process.

### 2. Data Relationship/Distribution Analysis
- **Objective**: Understand the distribution and relationships within the data.
- **Steps**:
  - Provide a Frequency table and plot to visualize Pickup counts by gender.
  - Provide Frequency tables and plots to visualize the distribution of Daily Average Minutes.
  - Analyze the relationship between:
    - Participant’s age and their Response time on singleton visual search.
    - Participant’s gender and their Response time on conjunction visual search.

### 3. Correlation Check
- **Objective**: Examine relationships between variables.
- **Steps**:
  - Produce a bivariate correlation table between Age, STAI, BRIEF_Total, DailyAvgMins, and VS_RT_correct_Single.

### 4. Linear Regression
- **Objective**: Assess if the minutes a person uses their mobile device per day predicts their visual search reaction time.
- **Steps**:
  - Perform a linear regression analysis.

### 5. Multiple Regression
- **Objective**: Evaluate the combined effect of multiple predictors on the outcome.
- **Steps**:
  - Add predictors (Age, Gender, Number of device pickups, etc.) to the regression model.
  - Determine if the variance accounted for in the outcome increases and if daily minutes of mbile usage remains a significant predictor.

### 6. Scenario 1 Analysis
- **Objective**: Test a hypothesis related to mobile usage and visual search performance.
- **Scenario**:
  - Participants were grouped by age and mobile usage.
  - They were asked to locate a target (red apple) among distractors (blue apples), and their reaction times were recorded.
- **Steps**:
  - Group participants and choose an appropriate Omnibus test statistic to test the hypothesis.
  - Justify the choice of test.
  - List assumptions and corresponding statistical tests.
  - Check and validate assumptions with visual charts.
  - Apply follow-on tests to identify specific effects.

### 7. Scenario 2 Analysis
- **Objective**: Test a hypothesis using a transformed dataset.
- **Scenario**:
  - Participants were asked to locate a target (red apple) among different distractors before and after a brain training exercise.
  - Their mobile usage was recorded and categorized.
- **Steps**:
  - Create groups and choose an appropriate Omnibus test statistic to test the hypothesis.
  - Justify the choice of test.
  - List assumptions and corresponding statistical tests.
  - Check and validate assumptions with visual charts.
  - Apply follow-on tests to identify specific effects.

## Instructions for Running the Analysis

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/Mclevesluna/Mobile-usage-and-search-performance.git
    cd your-repository-name
    ```

2. **Install Dependencies**:
    ```
    pip install -r requirements.txt
    ```
    Note: A GPU is not required for this project. It was developed on MacOS, ensuring compatibility with this operating system. However, it should work on other operating systems as well, though minor adjustments might be necessary.

Libraries used: 

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


3. **Run the Analysis**:
    Open the Jupyter notebook and run the cells to execute the analysis.

## Conclusion

This project demonstrates a comprehensive analysis of the relationship between mobile usage and visual search performance, accounting for various demographic variables and testing specific hypotheses using appropriate statistical methods.

## Acknowledgments

Special thanks to the contributors and the data providers.
