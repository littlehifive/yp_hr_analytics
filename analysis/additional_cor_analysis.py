#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 11:05:54 2023

@author: Zezhen Wu
"""
import pandas as pd
import sklearn
import joblib
import statsmodels.api as sm

data_path = '/Users/michaelfive/Desktop/R Directory/Youth Impact/yp_hr_analytics/myapp/data/'

# Import the cleaned and imputed dataset
df = pd.read_csv(data_path + 'application_all_mi.csv')
df = df.drop(columns = 'yi_hired')

# Get choice options for the features from the training dataset
choices = {f"choices_{col}": dat[col].unique().tolist() for col in dat.columns}

# Import the best fit xgboost model
xgb_fit_hiring = joblib.load(data_path + 'xgb_fit.pkl')
xgb_fit_perform = joblib.load(data_path + 'hired_xgb_fit.pkl')

# Retrieve the best estimator from the RandomizedSearchCV
best_model_hiring = xgb_fit_hiring.best_estimator_
best_model_perform = xgb_fit_perform.best_estimator_

# Extract the preprocessor from the best estimator
preprocessor_hiring = best_model_hiring.named_steps['preprocess']
preprocessor_perform = best_model_perform.named_steps['preprocess']

# ---- Predicting hiring ---- 

# Preprocess the single row using the fitted preprocessor
preprocessed_data_hiring = preprocessor_hiring.transform(df)
                
# Get the predicted probabilities for the preprocessed row
predicted_probabilities_hiring = best_model_hiring.named_steps['model'].predict_proba(preprocessed_data_hiring)
                
# Extract the probability for class 1 (hired)
probability_hired = predicted_probabilities_hiring[:, 1]  # The second value in the array
        
# Add the hiring probabilities to the dataframe
df['pred_hiring'] = probability_hired
        
# ---- Predicting levels gain ---- 
                
# Preprocess the single row using the fitted preprocessor
preprocessed_data_perform = preprocessor_perform.transform(df)
                
# Get the predicted probabilities for the preprocessed row
predicted_values_perform = best_model_perform.named_steps['model'].predict(preprocessed_data_perform)
        
# Add the hiring probabilities to the dataframe
df['pred_perform'] = predicted_values_perform

# ---- Get correlation between pred_hiring and pred_performance ---- 
print(df['pred_hiring'].corr(df['pred_perform']))
# correlation is 0.38
