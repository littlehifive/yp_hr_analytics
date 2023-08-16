# ConnectEd HR Analytics

## 1. What are the aims of this HR analytic task?

- Train models to predict current facilitator hiring decisions for ConnectEd

- Train models to predict current facilitator performance (i.e., average student levels gain) during implementation

- Use the trained models to predict future hire's likelihood of being hired and their performance

## 2. The most important features predicting hiring and performance

### For predicting hiring, the top 7 most important features are:
  - Total composite hiring score
  - Being hired in T3 2023
  - Highest education qualification
  - Not returning to Youth Impact
  - Working at an education organization
  - Being hired in T1 2020
  - Plan together with fellow facilitators if supervisor is not present

### For predicting performance, the top 7 most important features are:
  - Not returning to Youth Impact
  - Being hired in South East
  - Total composite hiring score
  - Volunteer to call the students and tutor them when coworker is sick
  - Highest education qualification
  - Interest in working with children 8-12
  - Plan individually if supervisor is not present

**Note: These features come from the best fit models and may not bear theoretical importance. For understanding how individual parameters predict the outcomes, please see `analysis/additional_pred_analysis.Rmd`.**

## 3. Guidelines to use this website for future ConnectEd hiring

- You can manually change any hiring information on the left, hit 'Generate hiring score', and see two predicted values.

- The predicted likelihood of being hired and the predicted average levels gain among students come from machine learning models (i.e., XGBoost) using data from T1 2020, T2 2022, T3 2023.

- Due to the uniqueness of the trained model, missing values are allowed for any variable when predicting future hiring or performance.

- DO NOT expect the predict values to change linearly according different input values of a given variable. They may not change at all if you only change the input value of one particular variable. XGBoost is a complex ensemble model, and the predicted values are a non-linear combination of many factors.

- You can also upload a csv file with using an existing template and download an updated csv with two columns with the predicted values at the end of the dataset.

## 4. Technical explanation

### Which model did we generate the predictions from?

Both the predicted likelihood of being hired and the predicted average levels gain among students are generated from a type of supervised learning model called XGBoost. Supervised learning, also known as supervised machine learning, is used to train algorithms to classify data or predict outcomes accurately.

XGBoost stands for "eXtreme Gradient Boosting." XGBoost is an ensemble method. It builds a series of decision trees, where each tree tries to correct the mistakes of the previous one. These trees are constructed in a sequence where each subsequent tree tries to minimize the error (or residual) from the trees before it. By combining the output of multiple 'weak' decision trees, it produces a strong predictive model.

We used XGBoost for two distinct tasks.

The first task is to use application survey information to predict existing hiring decisions by Youth Impact. Because the outcome variable is hiring (Yes/No), this task is considered to be a "classification problem."

The second task is to use application survey information to predict average levels gain among students that each hired facilitator was responsible for. Because the outcome variable is average levels gain (continuous), this task is considered to be a "regression problem."

### What are the general steps for building supervised learning models?

#### Data Preprocessing

- Multiple Imputation: We used the MICE package in R to conduct predictive mean matching and used the first imputed dataset for complete analysis.

- Standardizing Numerical Features: Scaling numerical features so that they have a range between 0 and 1 using the MinMaxScaler.

- Encoding Categorical Features: Converting categorical variables into a format that can be provided to machine learning algorithms to improve model accuracy, using one-hot encoding.

- Pipeline Creation for Data Transformation: Establishing two separate pipelines, one for numerical features and another for categorical ones. Then, combining both pipelines into a single preprocessing step.

#### Model Pipeline Construction

- Feature Processing & Model Training Pipeline: Creating a unified pipeline that first preprocesses the data (both numerical and categorical) and then feeds the processed data into a logistic regression model.

#### Hyperparameter Tuning

- Grid Search: Setting up a grid search to find the optimal hyperparameters for the model. This procedure will try out different combinations of hyperparameters and choose the best one based on model performance.

- Cross-Validation Scheme: Using K-Fold cross-validation (Repeated Stratified if predicting unbalanced binary outcomes) during the grid search. This approach ensures that the model is evaluated on different subsets of the training data multiple times, providing a more robust assessment of its performance.

#### Training the Model

- Fit the Model with Training Data (usually 80% of the original data): Using the optimal hyperparameters found during the grid search, training a variety of models (e.g., tree, random forest, xgboost, etc.) with the training dataset.

#### Prediction & Evaluation

- Evaluate on Test Data (usually 20% of the original data): Making predictions on the test data and comparing them to the actual outcomes.

## 5. Modeling procedure

### [Predicting Hiring](https://github.com/littlehifive/yp_hr_analytics/blob/main/analysis/hr_analytics.ipynb)

Because only 220 (11.5%) out of 1910 applicants were eventually hired, we faced an imbalanced classification problem. This resulted in a good predictions of the 0s (not hired), but not the 1s (hired), simply due to there are way fewer cases of hired facilitators to be predicted. Therefore, we adjusted the classification threshold based on the Receiver Operating Characteristic (ROC) curve in order to maximize the difference between the true positive rate (TPR) and the false positive rate (FPR). Additionally, we set class weight to be "balanced" in the models (i.e., the algorithm adjusts weights inversely proportional to class frequencies in the input data). We used repeated stratified k-fold cross-validation so that each set contains approximately the same percentage of samples of each target class as the complete set. We also used the balanced accuracy metric to take into account both false positives and false negatives and acquire a more informative view of the classifier's performance.

Out of all the models we attempted, the XGBoost model yielded the best balanced accuracy -- the arithmetic mean of sensitivity (true positive rate) and specificity (true negative rate). Because there are second stages of hiring where facilitators are trained and may not be hired due to their training performances, we priortize minimizing false negatives (hired but predicted to be not hired) over false positives (not hired but predicted to be hired), becasue we are only using information from the application survey from the first stage of hiring as predictors.

### [Predicting Performance](https://github.com/littlehifive/yp_hr_analytics/blob/main/analysis/hr_analytics_performance.ipynb)

We zoomed in on the hired facilitators for ConnectEd and attempted to predict the average learning levels gained by students that each facilitator was responsible for. Out of all the models we attempted, the XGBoost model yielded the smallest root mean squared error -- the difference between the values predicted by a model and the values actually observed.

## 6. Folder Structure

### `cleaning`:

  - `cleaning.py`: Python code for cleaning the application data
  
  - `merging.py`: Python code for merging the cleaned application data and doing name matching and merging with the entry data and student-level ConnectEd data
  
  - `functions.py`: Python code for utility functions for data cleaning
  
  - `name_matching.py`: Python code for fuzzy name matching between the application data and entry data
  
  - `imputation.R`: R code for multiple imputation of missing data using the `mice` R package (because `mice` is not readily available in Python)
  
  - `set_env.R`: `reticulate` is a package that allows you to run Python code in an R environment. I changed the virtual environment setting using this line of code because I have an M1 Macbook.

### `analysis`:

  - `feature_importances_hiring.png`: XGBoost feature importance plot predicting hiring
  
  - `feature_importances_performance.png`: XGBoost feature importance plot predicting performance 
  
  - `hr_analytics_performance.ipynb`: Python code for predicting performance in a Jupyter Notebook  
  
  - `hr_analytics.ipynb`: Python code for predicting hiring in a Jupyter Notebook

  - `analysis/additional_pred_analysis.Rmd` and `analysis/additional_pred_analysis.html`: R code for regression analysis (to interpret individual parameters and their p-values)

  - `analysis/additional_cor_analysis.py`: Python code for generating the predicted values for the two outcomes using a existing application dataset, and calculate their correlation (r = 0.38).
  
**Note**:
  
  - The easiest way to open and run Jupyter Notebook on your computer is to install [Anaconda](https://www.anaconda.com/). 
  
  - You may need to do `pip install <PACKAGE NAME>` (e.g. `pip install pandas`) *in Terminal* before you can run any code in the two notebooks.
  
  - If you wish to install the packages in the notebook without interacting with Terminal, you can create a code chunk and type `!pip install <PACKAGE NAME>` (e.g. `!pip install pandas`).
  
### `models`: `.pkl` is the file extension for saved machine learning models using `joblib.dump()` in Python.

  - `hired_svm_fit.pkl`: Support Vector Machine model predicting performance among hired facilitators
  
  - `hired_xgb_fit.pkl`: XGBoost model predicting performance among hired facilitators
  
  - `ML_models_predict_hiring.pkl`: All models predicting hiring among applicants
  
  - `xgb_fit.pkl`: XGBoost model predicting hiring among applicants
  
### `myapp`: A Python Shiny app built with the [`py-shiny` Python package](https://github.com/rstudio/py-shiny)

  - `app.py`: py-shiny code and html code for the HR Analytics Dashboard
  
  - `css`: css themes for the website downloaded from https://bootswatch.com/
  
  - `data`: de-identified data, images, and model files for the dashboard

**Note:** 

  - The app has not been deployed yet, pending future requests.
  
  - To run the app locally, open the `yp_hr_analytics.Rproj` in Rstudio, go to "myapp/app.py" and click "Run App".
  
  - Make sure you have all the necessary packages listed at the beginning of "app.py" installed (using `pip install`) before you run the app.
