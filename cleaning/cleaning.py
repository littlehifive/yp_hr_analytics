#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Data import and cleaning for HR analytics

@author: Zezhen Wu
'''

import os
import glob
import pandas as pd
import numpy as np
import functions

# global file path
path_application = '/Users/michaelfive/Dropbox/R&I/18_Facilitator_Database/02_Data_Analysis/100_HR-Analytics/01_connected_data/application_data/'
path_entry = '/Users/michaelfive/Dropbox/R&I/18_Facilitator_Database/02_Data_Analysis/100_HR-Analytics/01_connected_data/entry_data/'

# load the unified variablen names to replace the ones in the Excel sheets
df_var_name = pd.read_excel(path_application + 'variable_rename.xlsx')


# --------- 1. Application T1 2020 ---------

# load data
# dat_app_2020_raw = pd.read_excel(path_application + 'Facilitator_Applicants_Term1_2020.xlsx', sheet_name = 1)
dat_app_2020 = pd.read_excel(path_application + 'Facilitator_Applicants_Term1_2020.xlsx', sheet_name = 1)

# drop the region variables
cols_to_drop = [dat_app_2020.columns[0]] + list(dat_app_2020.columns[2:22])
dat_app_2020 = dat_app_2020.drop(cols_to_drop, axis=1)

# drop the two unamed columns at the end
dat_app_2020 = dat_app_2020.iloc[:, :-2]

# replace the column names with the correct ones
dat_app_2020.columns = df_var_name['T1_2020_new'].dropna()

# convert variable to correct types
dat_app_2020['contact_phone_2'] = dat_app_2020['contact_phone_2'].fillna(0).astype('int64').astype('str')
dat_app_2020['contact_phone_2'] = dat_app_2020['contact_phone_2'].replace('0', np.nan)

# drop variables not to be used in ML models (do not drop for now)
# dat_app_2020 = dat_app_2020.drop(['score_total', 'contact_phone_1', 'contact_phone_2', 'contact_email', 'emp_org'], axis = 1)

# convert years/months in to months
cols_to_convert = ['emp_duration', 'exp_length_teacher', 
                   'exp_length_school', 'exp_length_employee', 'exp_length_volunteer']

for col in cols_to_convert:
    dat_app_2020[col] = dat_app_2020[col].apply(functions.convert_duration_to_months)

# create experience dummy variables (to be in line with 2022 T2)
dat_app_2020 = functions.create_dummy_var(dat_app_2020, 'exp_length_teacher')
dat_app_2020 = functions.create_dummy_var(dat_app_2020, 'exp_length_employee')

# recode interest_w_child
dat_app_2020['interest_w_child'] = dat_app_2020['interest_w_child'].replace({'Yes': 4, 'No': 1})

# recode prac_scenario_behav
dat_app_2020['prac_scenario_behav'] = dat_app_2020['prac_scenario_behav'].replace(functions.likert_5_dict)

# recode prac_scenario_volunt
dat_app_2020['prac_scenario_volunt'] = dat_app_2020['prac_scenario_volunt'].apply(lambda x: 1 if x == 'Volunteer to teach the class' else 0)

# recode prac_scenario_behav
dat_app_2020['prac_scenario_noshow'] = dat_app_2020['prac_scenario_noshow'].replace(functions.prac_scenario_noshow_dict)

# add leading 0s to Omang IDs shorter than 9 digits
dat_app_2020['dem_omang'] = dat_app_2020['dem_omang'].astype(str).apply(functions.format_omang)

# export the dataset
dat_app_2020.to_csv(path_application + 'cleaned/application_2020_T1.csv')


# dat_app_2020.prac_scenario_noshow.value_counts()
# dat_app_2020.dem_omang.unique() 
# dat_app_2020['dem_omang'].astype(str).apply(len).value_counts()


# --------- 2. Application T2 2022 ---------
# load data
# dat_app_2022_raw = pd.read_excel(path_application + 'Facilitator_Applicants_Term2_2022_26042022.xlsx', sheet_name = 3)
dat_app_2022 = pd.read_excel(path_application + 'Facilitator_Applicants_Term2_2022_26042022.xlsx', sheet_name = 3)

# replace the column names with the correct ones
dat_app_2022.columns = df_var_name['T2_2022_new'].dropna()

# remove the first row (which contains the variable descriptions)
dat_app_2022 = dat_app_2022.iloc[1:]

# drop unused variables
dat_app_2022 = dat_app_2022.drop(['drop'], axis=1)

# filter out the zones facilitators
dat_app_2022 = dat_app_2022.loc[dat_app_2022.program_applied != 'Zones']

# recode gender
dat_app_2022['dem_gender'] = dat_app_2022['dem_gender'].replace(functions.gender_dict)

# recode interest_w_child
dat_app_2022['interest_w_child'] = dat_app_2022['interest_w_child'].replace(functions.likert_4_dict)

# experience variables are dichotomized instead of in months, thus renaming them
dat_app_2022 = dat_app_2022.rename(columns={'exp_length_teacher': 'exp_length_teacher_dummy', 'exp_length_employee': 'exp_length_employee_dummy'})

# recode prac_scenario_behav
dat_app_2022['prac_scenario_behav'] = dat_app_2022['prac_scenario_behav'].replace(functions.likert_5_dict)

# recode prac_scenario_volunt
dat_app_2022['prac_scenario_volunt'] = dat_app_2022['prac_scenario_volunt'].apply(lambda x: 1 if x == 'Volunteer to teach the class' else 0)

# recode prac_scenario_behav
dat_app_2022['prac_scenario_noshow'] = dat_app_2022['prac_scenario_noshow'].replace(functions.prac_scenario_noshow_dict)

# add leading 0s to Omang IDs shorter than 9 digits
dat_app_2022['dem_omang'] = dat_app_2022['dem_omang'].astype(str).apply(functions.format_omang)

# convert numeric columns to numeric
columns_to_convert = ['score_total', 'score_total_percentile', 'dem_age']  # Replace these with your column names

for column in columns_to_convert:
    dat_app_2022[column] = dat_app_2022[column].astype('float')

# export the dataset
dat_app_2022.to_csv(path_application + 'cleaned/application_2022_T2.csv')

# for col in dat_app_2022.columns:
#    print(f'\nValue counts for {col}:')
#    print(dat_app_2022[col].value_counts())

# --------- 3. Application T2 2023 ---------
# load data
# dat_app_2022_raw = pd.read_excel(path_application + 'Facilitator_Applicants_Term2_2022_26042022.xlsx', sheet_name = 3)
dat_app_2023 = pd.read_excel(path_application + 'ConnectEd_application_v20230522.xlsx', sheet_name = 2)

# replace the column names with the correct ones
dat_app_2023.columns = df_var_name['T2_2023_new'].dropna()

# drop unused variables
dat_app_2023 = dat_app_2023.drop(['drop'], axis=1)

# convert variable to correct types
dat_app_2023['contact_phone_1'] = dat_app_2023['contact_phone_1'].astype('str')
dat_app_2023['contact_phone_2'] = dat_app_2023['contact_phone_2'].replace('.', np.nan)

# create experience dummy variables (to be in line with 2022 T2)
dat_app_2023 = functions.create_dummy_var(dat_app_2023, 'exp_length_teacher')
dat_app_2023 = functions.create_dummy_var(dat_app_2023, 'exp_length_employee')

# recode prac_scenario_behav
dat_app_2023['prac_scenario_behav'] = dat_app_2023['prac_scenario_behav'].replace(functions.likert_5_dict)

# recode prac_scenario_volunt
dat_app_2023['prac_scenario_volunt'] = dat_app_2023['prac_scenario_volunt'].apply(lambda x: 1 if x == 'Volunteer to call the students and tutor them' else 0)

# recode prac_scenario_behav
dat_app_2023['prac_scenario_noshow'] = dat_app_2023['prac_scenario_noshow'].replace(functions.prac_scenario_noshow_dict)

# add leading 0s to Omang IDs shorter than 9 digits
dat_app_2023['dem_omang'] = dat_app_2023['dem_omang'].astype(str).apply(functions.format_omang)

# recode loc_region
dat_app_2023['return_to_yp'] = dat_app_2023['return_to_yp'].replace({'Return to YI': 'Yes', 'New': 'No'})

# export the dataset
dat_app_2023.to_csv(path_application + 'cleaned/application_2023_T2.csv')


# for col in dat_app_2023.columns:
#    print(f'\nValue counts for {col}:')
#    print(dat_app_2023[col].value_counts())


# --------- 4. Getting entry survey facilitator id and omang id ---------
df = pd.read_stata(path_entry + '02_entry_survey_clean_vConnectEd.dta')
functions.get_entry_id(df = df, omang = 'fac_omang', fac_id = 'fac_id')


# Directory where your .dta files are located
path_entry = '/Users/michaelfive/Dropbox/R&I/18_Facilitator_Database/02_Data_Analysis/100_HR-Analytics/01_connected_data/entry_data/'

# Lists of omang and facilitator variable names
omang_ids = ['fac_omang', 'fac_omang', 'fac_omang',
             'fac_omang', 'dem_omang',
             'fac_omang', 'dem_omang']  
facilitator_ids = ['fac_id', 'fac_id', 'fac_id',
                   'fac_id', 'dem_fac_id',
                   'fac_id', 'dem_fac_id']  
dataframes = []

# Get sorted list of .dta files
files = sorted([f for f in os.listdir(path_entry) if f.endswith('.dta')])
for file in ['2_entry_facilitators.dta', 'clean_entry_gov&new.dta']:
    files.remove(file) # does not have matched facilitator id with omang id

# Iterate over every file in the directory
for i, filename in enumerate(files):
    # Load .dta file into a DataFrame
    df = pd.read_stata(os.path.join(path_entry, filename))
    
    # Clean the ids and append the result to the list
    dataframes.append(functions.get_entry_id(df, omang_ids[i], facilitator_ids[i]))

# Concatenate all dataframes and remove duplicates
df_combined = pd.concat(dataframes).drop_duplicates()

# Rows where 'dem_omang' is not a 9-digit number
# not_nine_digits = df_combined[~df_combined['dem_omang'].astype(str).str.match('^\d{9}$')]

df_combined.to_csv(path_application + 'cleaned/entry_ids.csv',
                   index = False)

