#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fuzzy name matching for application data

@author: michaelfive
"""

# import pandas
import pandas as pd
from fuzzywuzzy import fuzz, process

# global file path
path_application = '/Users/michaelfive/Dropbox/R&I/18_Facilitator_Database/02_Data_Analysis/100_HR-Analytics/01_connected_data/application_data/'

# --------- 1.  load relevant datasets --------- 
dat_stu = pd.read_stata("/Users/michaelfive/Dropbox/R&I/18_Facilitator_Database/02_Data_Analysis/100_HR-Analytics/01_connected_data/0_pooled_56789.dta")

dat_entry_id = pd.read_csv(path_application + 'cleaned/entry_ids.csv',
                           dtype = {'facilitator_id_i': str,
                                    'dem_omang': str})

dat_entry_id['facilitator_id_i'] = dat_entry_id['facilitator_id_i'].str.split('.').str[0]

dat_app_all = pd.read_csv(path_application + 'cleaned/application_all.csv')
dat_app_all = dat_app_all.merge(dat_entry_id, on = 'dem_omang', how = 'left')

# --------- 2. create temporary datasets for checking purposes --------- 
# ids that are in the application data
a = set(dat_app_all['facilitator_id_i'].unique())

# ids that in the student data
b = set(dat_stu['facilitator_id_i'].unique())

# ids that are in the student data but not the application data
difference = b - a
print(difference)

# The problem is although the ids and the corresponding omang ids can be found
# in the entry survey, the omany ids cannot be found in the application data.
temp1 = dat_entry_id[dat_entry_id['facilitator_id_i'].isin(difference)]

temp2 = dat_stu[dat_stu['facilitator_id_i'].isin(difference)]
temp2 = temp2[['facilitator_i', 'facilitator_id_i']].drop_duplicates()

temp = temp1.merge(temp2, on = 'facilitator_id_i', how = 'left')

# temp has the correct id, dem_omang, and facilitator's name, but not matched 
# with the application data
temp.to_csv(path_application + 'id_manual_check.csv')

# we need to figure out if these facilitators actually are present in the application data
temp_app = dat_app_all[['dem_surname', 'dem_firstname', 'name', 'dem_omang']].drop_duplicates()

temp_app['name'] = temp_app.apply(
    lambda row: row['dem_firstname'] + ' ' + row['dem_surname'] if pd.isnull(row['name']) else row['name'],
    axis=1
)

temp_app = temp_app[['name', 'dem_omang']]

temp_app.to_csv(path_application + 'id_manual_check_application.csv')

# --------- 3. fuzzy matching by code --------- 
matches = []

# For each name in df1, find the closest match in df2
for name_correct in temp['facilitator_i']:
    # Using extractOne to find the single best match
    match = process.extractOne(name_correct, temp_app['name'])
    matches.append((name_correct, match[0], match[1]))  # (name1, matched_name, score)

matched_df = pd.DataFrame(matches, columns=['name_correct', 'best_match_in_application_data', 'matching_score'])

print(matched_df)

# --- CONCLUSION: ----
# Only one clear match because of mismatches in omang_id
# but it is not clear why the other names are not in the application data
# maybe they were hired for other purposes and ended up working with Connected
# or they did not fill in the application survey, but only the entry survey
