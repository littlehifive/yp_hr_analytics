#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for data cleaning

@author: Zezhen Wu
"""
import pandas as pd
import numpy as np

# Function that converts emp_duration to months
def convert_duration_to_months(duration):
    if pd.isnull(duration):
        return np.nan
    elif isinstance(duration, str):
        if '<1 month' in duration:
            return 1
        elif 'month' in duration:
            # strip ' month(s)' from the string and convert to float
            return float(duration.replace(' month(s)', ''))
        elif 'year' in duration:
            # strip ' years' from the string, convert to float, and multiply by 12
            return float(duration.replace(' years', '')) * 12


# Create experience dummy variables 
def create_dummy_var(df, col_name):
    new_col_name = col_name + '_dummy'
    conditions = [
        (df[col_name] < 12),
        (df[col_name] >= 12) & (df[col_name] <= 24),
        (df[col_name] > 24)
    ]
    choices = ['Less than 1 year', '1-2 years', 'More than 2 years']
    df[new_col_name] = np.select(conditions, choices, default = None)
    return df


# Function that adds leading 0s for Omang IDs
def format_omang(omang):
    if len(omang) == 8:
        return '0' + omang
    elif len(omang) == 7:
        return '00' + omang
    elif omang == "nan":
        return np.nan
    else:
        return omang
    
    
# Function that extract 9-digit Omang IDs from the entry data and convert them to a list    
def list_omang(df, omang):
    df_omang = df[omang].astype(str).apply(format_omang)
    df_omang = df_omang[df_omang.astype(str).str.match('^\d{9}$')].tolist()
    return df_omang

# Function that gets the facilitator and omang id from entry surveys
def get_entry_id(df, omang, fac_id):
    df['dem_omang'] = df[omang].astype(str).apply(format_omang)
    df['facilitator_id_i'] = df[fac_id]
    df = df[['facilitator_id_i', 'dem_omang']]
    return df
    

# 5-point recode dictionary
likert_5_dict = {
    'Strongly Disagree': 1,
    'Disagree': 2,
    'Neutral': 3,
    'Agree': 4,
    'Strongly Agree': 5
} 

# 4-point recode dictionary
likert_4_dict = {
    'Slightly': 1,
    'Moderately': 2,
    'Very': 3,
    'Extremely': 4
}       

# gender pronoun recode dictionary
gender_dict = {
    'Her': 'Female',
    'Him': 'Male',
    'They': 'Other',
    'Prefer not to say': 'Other'
} 

# prac_scenario_noshow recode dictionary
prac_scenario_noshow_dict = {
    'Gather fellow facilitators and start planning together': 'Planning together with fellow facilitators',
    'Gather fellow facilitators and make a plan for the day': 'Planning together with fellow facilitators',
    'Gather fellow facilitators and make a plan together': 'Planning together with fellow facilitators',

    'Find a quiet place to start planning individually': 'Plan individually',
    'Head to my classroom and start planning my lesson': 'Plan individually',
    'Read the calling guidlines and start planning my lesson': 'Plan individually',
    
    'Wait for your supervisor to return': 'Wait for supervisor or further instruction',
    'Stay home and await further instruction': 'Wait for supervisor or further instruction'
} 