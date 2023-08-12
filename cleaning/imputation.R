library(tidyverse)

# global file path
path_application <- '/Users/michaelfive/Dropbox/R&I/18_Facilitator_Database/02_Data_Analysis/100_HR-Analytics/01_connected_data/application_data/'

# 1. Impute the dataset for predicting hiring -----------------------------

# import the cleaned dataset
dat_app_all <- read_csv(file.path(path_application, 'cleaned/application_all.csv'))

# select variables for prediction models
selected_vars <- c('app_data_source', 
                  'loc_region', 'return_to_yp', "score_total",
                 'dem_gender',
                 'dem_age', 'edu_qual_highest', 'edu_qual_highest_year', 
                 'emp_yes', 'emp_org_type', 'emp_duration', 
                 'emp_currently', 'exp_w_child', 'exp_length_teacher', 
                 'exp_length_school', 'exp_length_employee',
                 'exp_length_volunteer', 'interest_w_child',
                 'prac_scenario_volunt', 'prac_scenario_noshow',
                 "yi_hired")

dat_app_s <- dat_app_all[, selected_vars]

# change strings to factor variables to be recognized by MICE
dat_app_s <- dat_app_s |> 
  mutate_if(is.character, as.factor)

# set seed for Multiple Imputation
set.seed(1234)

# define methods of imputation for each variable in the dataset
imp_methods <- c("", "sample", "sample", rep("pmm", 18))

dat_app_mi <- mice::mice(dat_app_s, method = imp_methods, m = 5)

# save the first imputed dataset (justifiable for classification purposes because we are not so interested in the uncertainty of parameters)
dat_app_mi_complete <- mice::complete(dat_app_mi, 1L)

# export dataset
write_csv(dat_app_mi_complete, file.path(path_application, "cleaned/application_all_mi.csv"))

# 2. Impute the dataset for predicting performance -----------------------------

# import the cleaned dataset
dat_app_hired <- read_csv(file.path(path_application, 'cleaned/application_hired.csv'))

# select variables for prediction models
selected_vars <- c('app_data_source', 
                   'loc_region', 'return_to_yp', "score_total",
                   'dem_gender',
                   'dem_age', 'edu_qual_highest', 'edu_qual_highest_year', 
                   'emp_yes', 'emp_org_type', 'emp_duration', 
                   'emp_currently', 'exp_w_child', 'exp_length_teacher', 
                   'exp_length_school', 'exp_length_employee',
                   'exp_length_volunteer', 'interest_w_child',
                   'prac_scenario_volunt', 'prac_scenario_noshow',
                   "level_gains")

dat_app_hired_s <- dat_app_hired[, selected_vars]

# change strings to factor variables to be recognized by MICE
dat_app_hired_s <- dat_app_hired_s |> 
  mutate_if(is.character, as.factor)

# set seed for Multiple Imputation
set.seed(1234)

# define methods of imputation for each variable in the dataset
imp_methods <- c("", "sample", "sample", rep("pmm", 18))

dat_app_hired_mi <- mice::mice(dat_app_hired_s, method = imp_methods, m = 5)

# save the first imputed dataset (justifiable for classification purposes because we are not so interested in the uncertainty of parameters)
dat_app_hired_mi_complete <- mice::complete(dat_app_hired_mi, 1L)

# export dataset
write_csv(dat_app_hired_mi_complete, file.path(path_application, "cleaned/application_hired_mi.csv"))
