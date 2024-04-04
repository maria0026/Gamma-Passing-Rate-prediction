import os
import pandas as pd
import prepare_df

data_folder='/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/data_final'
extension = 'dcm'
first_keyword = 'Predicted'
second_keyword='Portal'

reference_folder = '/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/reference'
evaluation_folder = '/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/evaluation'
txt_folder = '/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/txt'

prepare_df.data_sorting(data_folder, extension, first_keyword, second_keyword)
files_without_pair=prepare_df.find_dcm_without_pair(reference_folder, evaluation_folder)
print("Dcm without pair", len(files_without_pair))
#there are some files (825) without a pair, we will not take them into account

df=prepare_df.create_df(reference_folder, evaluation_folder, txt_folder)
print(df)
df[['ref', 'eval', 'txt']].to_csv('ref_eval2.csv', sep='\t', index=False)
