import pandas as pd
import numpy as np
from skimpy import clean_columns
import functions

# global file path
path_application = '/Users/michaelfive/Dropbox/R&I/18_Facilitator_Database/02_Data_Analysis/100_HR-Analytics/01_connected_data/application_data/'

# --------- 1. Merge application data ---------

dat_app_2020 = pd.read_csv(path_application + 'cleaned/application_2020_T1.csv', dtype={'contact_phone_2': str})
dat_app_2022 = pd.read_csv(path_application + 'cleaned/application_2022_T2.csv', dtype={'contact_phone_2': str})
dat_app_2023 = pd.read_csv(path_application + 'cleaned/application_2023_T2.csv', dtype={'contact_phone_2': str})


# Add the 'app_data_source' column to each dataframe
dat_app_2020['app_data_source'] = 'T1 2020'
dat_app_2022['app_data_source'] = 'T2 2022'
dat_app_2023['app_data_source'] = 'T3 2023'

dat_app_all = pd.concat([dat_app_2020, dat_app_2022, dat_app_2023], 
                        ignore_index = True, sort = False)

# Move the 'app_data_source' column to the first position
cols = list(dat_app_all.columns)
cols.insert(0, cols.pop(cols.index('app_data_source')))
cols.remove('Unnamed: 0')

dat_app_all = dat_app_all[cols]

# NOT merging with facilitator database (because some facilitators may have left when the facilitator database was created in 2022)
# dat_fac = pd.read_excel(path_application + "facilitator_database_view_R&I.xlsx")

# dat_fac = clean_columns(dat_fac, case='snake')

# dat_fac["dem_omang"] = dat_fac['omang'].astype(str).apply(functions.format_omang)

# dat_fac = dat_fac[['dem_omang', 'id']]

# # export the dataset
# dat_fac.to_csv(path_application + 'cleaned/facilitator_id.csv')

# --------- 2. Getting a list of hired facilitators from multiple entry surveys ---------

# Directory where the dta files are located
path_entry = '/Users/michaelfive/Dropbox/R&I/18_Facilitator_Database/02_Data_Analysis/100_HR-Analytics/01_connected_data/entry_data/'

# List of omang variable names for each dta file in the same order
omang_var_names = ['fac_omang',
                   'fac_omang',
                   'fac_omang',
                   'fac_omang',
                   'facilitator_omang',
                   'dem_omang',
                   'fac_omang',
                   'dem_omang',
                   'dem_omang']

# List to store all Omang IDs
all_omang_ids = []

# Get all dta files in the directory
file_list = sorted(glob.glob(os.path.join(path_entry, '*.dta')))

for file_path, omang_var in zip(file_list, omang_var_names):
    # Read the dta file
    df = pd.read_stata(file_path)
    
    # Extract the Omang IDs
    omang_ids = functions.list_omang(df, omang_var)
    
    # Append the list of Omang IDs to the main list
    all_omang_ids.extend(omang_ids)

# Remove duplicates by converting the list to a set and then back to a list
all_omang_ids = list(set(all_omang_ids))

# --------- 3. Create outcome variable (hired or not) and merge with facilitator id---------

# create outcome variable (whether hired by Youth Impact)
dat_app_all['dem_omang'] = dat_app_all['dem_omang'].astype(str)
dat_app_all['yi_hired'] = np.where(dat_app_all['dem_omang'].isin(all_omang_ids), 1, 0)

# export the dataset for predicting hiring
dat_app_all.to_csv(path_application + 'cleaned/application_all.csv')

dat_entry_id = pd.read_csv(path_application + 'cleaned/entry_ids.csv',
                           dtype = {'facilitator_id_b': str,
                                    'dem_omang': str})

dat_entry_id['facilitator_id_b'] = dat_entry_id['facilitator_id_b'].str.split('.').str[0]

dat_app_all =  dat_app_all.merge(dat_entry_id, on = 'dem_omang', how = 'left')

# --------- 4. Merge with connected data ---------
dat_stu = pd.read_stata("/Users/michaelfive/Dropbox/R&I/18_Facilitator_Database/02_Data_Analysis/100_HR-Analytics/01_connected_data/0_pooled_56789.dta")

# pending...
dat_stu_s = dat_stu.groupby('facilitator_id_b')['stud_age_b'].mean()
