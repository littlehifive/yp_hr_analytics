#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 09:23:43 2023

@author: Zezhen Wu
"""

# ------------------------ Import Packages ------------------------
from pathlib import Path
from shiny import App, ui, render, reactive
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ------------------------ Create Global Variables ------------------------

# Set file path
data_path = Path(__file__).parent/ "data"
css_file = Path(__file__).parent / "css" / "bootstrap.min.css"

# Import the cleaned and imputed dataset
dat = pd.read_csv(data_path / 'application_all_mi.csv')

# Get choice options for the features from the training dataset
choices = {f"choices_{col}": dat[col].unique().tolist() for col in dat.columns}

# Import the best fit xgboost model
xgb_fit_hiring = joblib.load(data_path / 'xgb_fit.pkl')
xgb_fit_perform = joblib.load(data_path / 'hired_xgb_fit.pkl')

# Import the feature importance plots
feature_importance_hiring = mpimg.imread(data_path / 'feature_importances_hiring.png')
feature_importance_perform = mpimg.imread(data_path / 'feature_importances_performance.png')

# Retrieve the best estimator from the RandomizedSearchCV
best_model_hiring = xgb_fit_hiring.best_estimator_
best_model_perform = xgb_fit_perform.best_estimator_

# Extract the preprocessor from the best estimator
preprocessor_hiring = best_model_hiring.named_steps['preprocess']
preprocessor_perform = best_model_perform.named_steps['preprocess']
        
# ------------------------ Part 1: Create UI ------------------------

# A card component wrapper function
def ui_card(title, *args):
    return (
        ui.div(
            {"class": "card mb-4"},
            ui.div(title, class_="card-header"),
            ui.div({"class": "card-body"}, *args),
        ),
    )

# Create UI    
app_ui = ui.page_fluid(
  
  # Include CSS file from https://bootswatch.com/
  ui.include_css(css_file),
  
  # Dashboard Title
  ui.row(ui.tags.h2("Connected HR Analytics Dashboard")),
    
  # Create row structure  
  ui.row(
    
  # ------------- First column of inputs -------------
  ui.column(2,
  ui.panel_well(
    ui.tags.h4("Hiring Information"), 
    ui.input_selectize("select_app_data_source", "Term and year for the hiring", choices['choices_app_data_source'] + ['After T3 2023']),
    ui.input_selectize("select_loc_region", "Region", sorted(choices['choices_loc_region'] + ['Other'])),
    ui.input_selectize("select_return_to_yp", "Returning to Youth Impact?", choices['choices_return_to_yp']),
    ui.input_selectize("select_emp_yes", "Ever employed before?", choices['choices_emp_yes']),
    ui.input_selectize("select_emp_org_type", "Industry of the organization you work at", sorted(choices['choices_emp_org_type'])),
    ui.input_selectize("select_emp_currently", "Currently employed at the organization?", choices['choices_emp_currently']),
    ui.input_numeric("select_emp_duration", "Number of months employed at the organization", round(np.median(dat["emp_duration"]))),
    ui.input_selectize("select_exp_w_child", "Any experience working with children?", choices['choices_exp_w_child']),
    ui.input_selectize("select_interest_w_child", "Interest in working with children 8-12 (on a scale of 1 (Not much) - 4 (Very))", list(range(1, 5)))
    )
  ),
   
  # ------------- Second column of inputs -------------  
    ui.column(2,
    ui.panel_well(
    ui.tags.h4(" "),
      ui.input_selectize("select_dem_gender", "Gender", choices['choices_dem_gender']),
      ui.input_numeric("select_dem_age", "Age", round(np.median(dat["dem_age"]))),
      ui.input_selectize("select_edu_qual_highest", "Highest education qualification", sorted(choices['choices_edu_qual_highest'])),
      ui.input_numeric("select_edu_qual_highest_year", "Year when highest education is completed ", round(np.median(dat["edu_qual_highest_year"]))),
      ui.input_numeric("select_exp_length_teacher", "Month spent working as a teacher", round(np.median(dat["exp_length_teacher"]))),
      ui.input_numeric("select_exp_length_school", "Month spent working as intern/assistant teacher/TSP at a school", round(np.median(dat["exp_length_school"]))),
      ui.input_numeric("select_exp_length_employee", "Month spent working as a full-time employee working w children", round(np.median(dat["exp_length_employee"]))),
      ui.input_numeric("select_exp_length_volunteer", "Month spent working as a volunteer/tutor for children", round(np.median(dat["exp_length_volunteer"]))),
      ui.input_selectize("select_prac_scenario_volunt", "Scenario question: Volunteer to call the students and tutor them when coworker is sick (1 (Yes), 0 (No))", [1, 0]),
      ui.input_selectize("select_prac_scenario_noshow", "Scenario question: What to do if supervisor not present", choices['choices_prac_scenario_noshow'])
      )
    ),
   
   # ------------- Results panel -------------
    ui.column(8,
    
    # ------------- First Navigation Bar for Prediction -------------
    ui.navset_tab(
      
      ui.nav("Generate predictions from selection",
      # Action button
      ui.input_action_button("generate", "Click to generate hiring scores:"),
      
      # Results
      ui.div(
              {"class": "card mb-3"},
              ui.div(
              {"class": "card-footer"},
              {"style": "font-weight: bold"},
              ui.output_text("result")
              )
            )
          ),
     
     # ------------- Second Navigation Bar for Prediction -------------
      ui.nav("Generate predictions from data files",
      
      # A card for uploading dataset
      ui_card(
      "Upload an application dataset",  
      ui.a("An example csv template can be found here on Dropbox.",
           href = "https://www.dropbox.com/scl/fi/8f0x6mep5tfxggcouug52/example_data.csv?rlkey=6i54tylt56hqsj14sm5s90dre&dl=0",
           target = "_blank"
          ),
      ui.p(""),
      ui.p("The two predicted scores will be automatically added to the uploaded dataset."),
      ui.p(""),
      # Upload button
      ui.input_file("csv_upload", "Choose a .csv file to upload:", accept=[".csv"], multiple=False),
      ui.output_ui("pd_uploaded"),
      ui.output_text("upload_check")
      ),
      
      # A card for downloading updated dataset, conditional on dataset being updated
      ui.panel_conditional(
      "output.upload_check == 'Scroll to the very right and you have generated the predicted scores!'",
      ui_card(
        "Download the updated application dataset",
        ui.p("You can download the updated application dataset with predicted scores to your local computer by clicking the button below."),
        ui.download_button("download_csv", "Download CSV"))
        )
      )
    ),  

     # ------------- Informational Tabs -------------
      ui.navset_tab(
      
      # Tab 1
      ui.nav("General Information",
      
      ui.tags.br(),
      ui.h5("1. What are the aims of this HR analytic task?"),
      ui.tags.br(),
      ui.tags.li("Train models to predict current facilitator hiring decisions for ConnectEd"),
      ui.tags.li("Train models to predict current facilitator performance (i.e., average student levels gain) during implementation"),
      ui.tags.li("Use the trained models to predict future hire's likelihood of being hired and their performance"),
      
      ui.tags.br(),
      ui.h5("2. The most important features predicting hiring and performance"),
      ui.tags.br(),
      ui.h6("For more information, see the Graphics tab."),
      ui.tags.br(),
      ui.h6("For predicting hiring, the top 7 most important features are:"),
      ui.tags.li("Total composite hiring score"),
      ui.tags.li("Being hired in T3 2023"),
      ui.tags.li("Highest education qualification"),
      ui.tags.li("Not returning to Youth Impact"),
      ui.tags.li("Working at an education organization"),
      ui.tags.li("Being hired in T1 2020"),
      ui.tags.li("Plan together with fellow facilitators if supervisor is not present"),
      ui.tags.br(),        
      ui.h6("For predicting performance, the top 7 most important features are:"),
      ui.tags.li("Not returning to Youth Impact"),
      ui.tags.li("Being hired in South East"),
      ui.tags.li("Total composite hiring score"),
      ui.tags.li("Volunteer to call the students and tutor them when coworker is sick"),
      ui.tags.li("Highest education qualification"),
      ui.tags.li("Interest in working with children 8-12"),
      ui.tags.li("Plan individually if supervisor is not present")
      ),
        
      # Tab 2
      ui.nav("How to use",
      ui.tags.br(),
      ui.h5("Guidelines to use this website for future ConnectEd hiring"),
      ui.tags.br(),
        ui.tags.li("You can manually change any hiring information on the left, hit 'Generate hiring score', and see two predicted values."),
      ui.tags.br(),
        ui.tags.li("The predicted likelihood of being hired and the predicted average levels gain among students come from machine learning models (i.e., XGBoost) using data from T1 2020, T2 2022, T3 2023."),
      ui.tags.br(),
        ui.tags.li("Due to the uniqueness of the trained model, missing values are allowed for any variable when predicting future hiring or performance."),
      ui.tags.br(),
        ui.tags.li("DO NOT expect the predict values to change linearly according different input values of a given variable. They may not change at all if you only change the input value of one particular variable. XGBoost is a complex ensemble model, and the predicted values are a non-linear combination of many factors."),
      ui.tags.br(),
        ui.tags.li("You can also upload a csv file with using an existing template and download an updated csv with two columns with the predicted values at the end of the dataset.")
      ),
      
      # Tab 3
      ui.nav("Technical Explanation", 
      ui.tags.br(),
      ui.h5("1. Which model did we generate the predictions from?"),
        ui.p(
            """
            Both the predicted likelihood of being hired and the predicted average levels gain among students are generated from a type of supervised learning model called XGBoost.
            Supervised learning, also known as supervised machine learning, is used to train algorithms to classify data or predict outcomes accurately.
            """
        ),
        ui.p(
          """
            XGBoost stands for "eXtreme Gradient Boosting." XGBoost is an ensemble method. It builds a series of decision trees, where each tree tries to correct the mistakes of the previous one. 
            These trees are constructed in a sequence where each subsequent tree tries to minimize the error (or residual) from the trees before it. 
            By combining the output of multiple 'weak' decision trees, it produces a strong predictive model.
          """
        ),
        ui.p(
          """
            We used XGBoost for two distinct tasks.
          """),
        ui.p(
          """
            The first task is to use application survey information to predict existing hiring decisions by Youth Impact. 
            Because the outcome variable is hiring (Yes/No), this task is considered to be a "classification problem."
          """
        ),
        ui.p(
          """
            The second task is to use application survey information to predict average levels gain among students that each hired facilitator was responsible for.
            Because the outcome variable is average levels gain (continuous), this task is considered to be a "regression problem."
          """
        ),
      ui.tags.br(),
      ui.h5("2. What are the general steps for building supervised learning models?"), 
        ui.tags.b("Data Preprocessing"),
        ui.tags.li("Multiple Imputation: We used the MICE package in R to conduct predictive mean matching and used the first imputed dataset for complete analysis."),
        ui.tags.li("Standardizing Numerical Features: Scaling numerical features so that they have a range between 0 and 1 using the MinMaxScaler."),
        ui.tags.li("Encoding Categorical Features: Converting categorical variables into a format that can be provided to machine learning algorithms to improve model accuracy, using one-hot encoding."),
        ui.tags.li("Pipeline Creation for Data Transformation: Establishing two separate pipelines, one for numerical features and another for categorical ones. Then, combining both pipelines into a single preprocessing step."),
        
        ui.tags.b("Model Pipeline Construction"),
        ui.tags.li("Feature Processing & Model Training Pipeline: Creating a unified pipeline that first preprocesses the data (both numerical and categorical) and then feeds the processed data into a logistic regression model."),
        
        ui.tags.b("Hyperparameter Tuning"),
        
        ui.tags.li("Grid Search: Setting up a grid search to find the optimal hyperparameters for the model. This procedure will try out different combinations of hyperparameters and choose the best one based on model performance."),
        ui.tags.li("Cross-Validation Scheme: Using K-Fold cross-validation (Repeated Stratified if predicting unbalanced binary outcomes) during the grid search. This approach ensures that the model is evaluated on different subsets of the training data multiple times, providing a more robust assessment of its performance."),
        
        ui.tags.b("Training the Model"),
        ui.tags.li("Fit the Model with Training Data (usually 80% of the original data): Using the optimal hyperparameters found during the grid search, training a variety of models (e.g., tree, random forest, xgboost, etc.) with the training dataset."),

        ui.tags.b("Prediction & Evaluation"),
        ui.tags.li("Evaluate on Test Data (usually 20% of the original data): Making predictions on the test data and comparing them to the actual outcomes.")
     
        ),
        
      # Tab 4 
      ui.nav("Modeling Procedure",
      ui.tags.br(),
      ui.h5("All analytical details can be found on the following two jupyter notebooks:"),
      ui.tags.br(),
        ui.a(
                "1. Predicting Hiring",
                href="https://github.com/littlehifive/yp_hr_analytics/blob/main/analysis/hr_analytics.ipynb",
                target="_blank",
            ),
        ui.p(
          """
            Because only 220 (11.5%) out of 1910 applicants were eventually hired, we faced an imbalanced classification problem.
            This resulted in a good predictions of the 0s (not hired), but not the 1s (hired), simply due to there are way fewer cases of hired facilitators to be predicted.
            Therefore, we adjusted the classification threshold based on the Receiver Operating Characteristic (ROC) curve in order to maximize the difference between the true positive rate (TPR) and the false positive rate (FPR).
            Additionally, we set class weight to be "balanced" in the models (i.e., the algorithm adjusts weights inversely proportional to class frequencies in the input data).
            We used repeated stratified k-fold cross-validation so that each set contains approximately the same percentage of samples of each target class as the complete set.
            We also used the balanced accuracy metric to take into account both false positives and false negatives and acquire a more informative view of the classifier's performance.
          """
          ),
        ui.p(
          """
          Out of all the models we attempted, the XGBoost model yielded the best balanced accuracy -- the arithmetic mean of sensitivity (true positive rate) and specificity (true negative rate).
          Because there are second stages of hiring where facilitators are trained and may not be hired due to their training performances, 
          we priortize minimizing false negatives (hired but predicted to be not hired) over false positives (not hired but predicted to be hired), becasue we are only using information from the application survey from the first stage of hiring as predictors.
          """
        ),  
        ui.a(
                "2. Predicting Performance",
                href="https://github.com/littlehifive/yp_hr_analytics/blob/main/analysis/hr_analytics_performance.ipynb",
                target="_blank",
            ),
        ui.p(
          """
          We zoomed in on the hired facilitators for ConnectEd and attempted to predict the average learning levels gained by students that each facilitator was responsible for.
          Out of all the models we attempted, the XGBoost model yielded the smallest root mean squared error -- the difference between the values predicted by a model and the values actually observed.
          """
        ),     
      ),
      
      # Tab 5
      ui.nav("Graphics",
      ui.tags.br(),
      ui.h5("Feature Importance Plot"),

        ui.p(
          """
          The feature importance plot for XGBoost visually displays the importance of each feature used in the model. Feature importance helps in understanding which features are the most influential in making predictions.
          """
        ),
        ui.p(
          """
          A feature with a high importance score played a more significant role in the model's decision-making process than features with lower scores.
          """
        ),
        ui.a(
                "For a comprehensive list of variables and their descriptions, see this spreadsheet on Dropbox.",
                href="https://www.dropbox.com/scl/fi/fztpcn8shtuuf52mv9vce/variable_rename.xlsx?rlkey=phy9a2b5rglexdr9jr396wm63&dl=0",
                target="_blank",
            ),        
        ui.output_plot("feature_importance_plot", width='100%', height='600px')
        )
      )
    )
  )
)

# ------------------------ Part 1: Create Server Functions ------------------------

def server(input, output, session):
    
    # ------------- Render text based on reactive values from variable selection -------------
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
        
        # ---- Predicting hiring ---- 

        # Preprocess the single row using the fitted preprocessor
        preprocessed_row_hiring = preprocessor_hiring.transform(df)
        
        # Get the predicted probabilities for the preprocessed row
        predicted_probabilities_hiring = best_model_hiring.named_steps['model'].predict_proba(preprocessed_row_hiring)
        
        # Extract the probability for class 1 (hired)
        probability_hired = predicted_probabilities_hiring[0][1]  # The second value in the array
        
        # ---- Predicting levels gain ---- 
        
        # Preprocess the single row using the fitted preprocessor
        preprocessed_row_perform = preprocessor_perform.transform(df)
        
        # Get the predicted probabilities for the preprocessed row
        predicted_value_perform = best_model_perform.named_steps['model'].predict(preprocessed_row_perform)
        
        # Predict the hiring probabilities for this new potential hire
        return ("For the new facilitator(s), "
                f"the predicted likelihood of being hired is {probability_hired * 100:.2f}%, "
                f"and the predicted average levels gain among students is {predicted_value_perform[0]:.2f}.")
    
    
    # ------------- Render variable importance plot -------------
    @output
    @render.plot
    def feature_importance_plot():
        # Create 1 row and 2 columns of subplots
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))  # Adjust figsize as needed

        # Display the images
        axes[0].imshow(feature_importance_hiring)
        axes[0].axis('off')  # Turn off axis for the first image
        axes[0].set_title('Features Predicting Hiring')  # Set title for the first image
        
        axes[1].imshow(feature_importance_perform)
        axes[1].axis('off')  # Turn off axis for the second image
        axes[1].set_title('Features Predicting Performance')  # Set title for the first image
        
        plt.tight_layout()
    
    # ------------- Create predictions in uploaded dataset -------------
    def get_csv():
        f: list[FileInfo] = input.csv_upload()
        df = pd.read_csv(f[0]["datapath"])
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
        
        return(df)

    # ------------- Render upload button -------------
    @output    
    @render.ui 
    def pd_uploaded():
        if input.csv_upload() is None:
            return "Please upload a csv file."
        df = get_csv()
        return ui.HTML('<div style="overflow: auto; max-height: 400px; width: 1200px;"> <style> table, th, td {border: 1px solid black;} table {border-collapse: collapse;} </style>' + df.to_html(classes="table table-striped") + '</div>')
    
    # ------------- Render text for uploading checks -------------
    @output    
    @render.text
    def upload_check():
        try:
            df = get_csv()
            if not df.empty:
                return "Scroll to the very right and you have generated the predicted scores!"
            else:
                return "Warning: You uploaded dataset is empty, please double check!"
        except Exception as e:
            return f"Warning: You haven't uploaded a dataset yet!"

    # ------------- Render download button -------------
    @session.download(filename="hr_analytics_connected_pred.csv")
    def download_csv():
        yield get_csv().to_csv()
    
# Combine UI and Server Function into a Shiny App
app = App(app_ui, server)
