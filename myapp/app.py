#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 09:23:43 2023

@author: michaelfive
"""

from shiny import App, ui, render, reactive
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb

# global file path
path_application = '/Users/michaelfive/Dropbox/R&I/18_Facilitator_Database/02_Data_Analysis/100_HR-Analytics/01_connected_data/application_data/'

# import the cleaned dataset
dat = pd.read_csv(path_application + 'cleaned/application_all_mi.csv')

# Get choice options for the features from the training dataset
choices = {f"choices_{col}": dat[col].unique().tolist() for col in dat.columns}

# import the best fit xgboost model
xgb_fit = joblib.load(path_application + 'xgb_fit.pkl')

# Part 1: ui ----
app_ui = ui.page_fluid(
  
  ui.row(
  ui.column(2,
    ui.input_selectize("select_app_data_source", "Term and year for the hiring", choices['choices_app_data_source'] + ['After T3 2023']),
    ui.input_selectize("select_loc_region", "Region", sorted(choices['choices_loc_region'] + ['Other'])),
    ui.input_selectize("select_return_to_yp", "Returning to Youth Impact?", choices['choices_return_to_yp']),
    ui.input_selectize("select_dem_gender", "Gender", choices['choices_dem_gender']),
    ui.input_selectize("select_edu_qual_highest", "Highest education qualification", sorted(choices['choices_edu_qual_highest'])),
    ui.input_selectize("select_emp_yes", "Ever employed before?", choices['choices_emp_yes']),
    ui.input_selectize("select_emp_org_type", "Industry of the organization you work at", sorted(choices['choices_emp_org_type'])),
    ui.input_selectize("select_emp_currently", "Currently employed at the organization?", choices['choices_emp_currently']),
    ui.input_selectize("select_exp_w_child", "Any experience working with children?", choices['choices_exp_w_child']),
    ui.input_selectize("select_prac_scenario_noshow", "Scenario question: What to do if supervisor not present", choices['choices_prac_scenario_noshow'])),
    
    ui.column(2,
      ui.input_numeric("select_dem_age", "Age", round(np.median(dat["dem_age"]))),
      ui.input_numeric("select_edu_qual_highest_year", "Year when highest education is completed ", round(np.median(dat["edu_qual_highest_year"]))),
      ui.input_numeric("select_emp_duration", "Number of months employed at the organization", round(np.median(dat["emp_duration"]))),
      ui.input_numeric("select_exp_length_teacher", "Month spent working as a teacher", round(np.median(dat["exp_length_teacher"]))),
      ui.input_numeric("select_exp_length_school", "Month spent working as intern/assistant teacher/TSP at a school", round(np.median(dat["exp_length_school"]))),
      ui.input_numeric("select_exp_length_employee", "Month spent working as a full-time employee working w children", round(np.median(dat["exp_length_employee"]))),
      ui.input_numeric("select_exp_length_volunteer", "Month spent working as a volunteer/tutor for children", round(np.median(dat["exp_length_volunteer"]))),
      ui.input_selectize("select_interest_w_child", "Interest in working with children 8-12 (on a scale of 1 (Not much) - 4 (Very))", list(range(1, 5))),
      ui.input_selectize("select_prac_scenario_volunt", "Scenario question: Volunteer to call the students and tutor them when coworker is sick (1 (Yes), 0 (No))", [1, 0])
    ),
    
    ui.column(8,
      ui.input_action_button("generate", "Generate hiring scrore:"),
      ui.output_text("result"))
    )
    
  
)

# Part 2: server ----
def server(input, output, session):
    @output
    @render.text
    @reactive.event(input.generate) # Take a dependency on the button
    def result():
        variables = {
          'app_data_source': "",
          'loc_region': input["select_loc_region"](),
          'return_to_yp': input["select_return_to_yp"](),
          'score_total': None,
          'dem_gender': input["select_dem_gender"](),
          'dem_age': input["select_dem_age"](),
          'edu_qual_highest': input["select_edu_qual_highest"](),
          'edu_qual_highest_year': input["select_edu_qual_highest_year"](),
          'emp_yes': input["select_emp_yes"](),
          'emp_org_type': input["select_emp_org_type"](),
          'emp_duration': input["select_emp_duration"](),
          'emp_currently': input["select_emp_currently"](),
          'exp_w_child': input["select_exp_w_child"](),
          'exp_length_teacher': input["select_exp_length_teacher"](),
          'exp_length_school': input["select_exp_length_school"](),
          'exp_length_employee': input["select_exp_length_employee"](),
          'exp_length_volunteer': input["select_exp_length_volunteer"](),
          'interest_w_child': input["select_interest_w_child"](),
          'prac_scenario_volunt': input["select_prac_scenario_volunt"](),
          'prac_scenario_noshow': input["select_prac_scenario_noshow"]()}
        
        df = pd.DataFrame([variables])
        
        # Retrieve the best estimator from the RandomizedSearchCV
        best_model = xgb_fit.best_estimator_

        # Extract the preprocessor from the best estimator
        preprocessor = best_model.named_steps['preprocess']
        
        # Preprocess the single row using the fitted preprocessor
        preprocessed_row = preprocessor.transform(df)
        
        # Get the predicted probabilities for the preprocessed row
        predicted_probabilities = best_model.named_steps['model'].predict_proba(preprocessed_row)
        
        # Extract the probability for class 1 (hired)
        probability_hired = predicted_probabilities[0][1]  # The second value in the array
        
        return f"Predicted likelihood of being hired: {probability_hired * 100:.2f}%"

# Combine into a shiny app.
# Note that the variable must be "app".
app = App(app_ui, server)
