import os
import pandas as pd
import prepare_df
import gamma
import preprocess 
from sklearn.model_selection import train_test_split
import model_gamma
 

#prepraring df
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
df[['ref', 'eval', 'txt']].to_csv('ref_eval3.csv', sep=';', index=False)
df=pd.read_csv('ref_eval3.csv', sep=';')
print(df)

#gamma for df
#df = pd.read_csv('ref_eval2.csv', sep=';')
#print(df)
dose_percent_threshold=2
distance_mm_threshold=2
lower_percent_dose_cutoff=5
#df=gamma.calculate_gamma_for_df(df, dose_percent_threshold=dose_percent_threshold, distance_mm_threshold=distance_mm_threshold, lower_percent_dose_cutoff=lower_percent_dose_cutoff, draw=False)
#df[['pass_ratio','gamma_txt']].to_csv('gamma_results5.csv', sep='\t', index=False)


#load and preprocess images
df = pd.read_csv('data_for_nn5.csv', sep=',')
print(df)
df = df.dropna()
df=df[df['pass_ratio']>90]
print(df)
df['class'] = df['gamma_txt'].apply(preprocess.map_to_class)
#print(df['class'])

df_train, df_test= train_test_split(df, test_size=0.2, random_state=42)

ref_desired_size=(1024,1024)
eval_desired_size=(1190, 1190)
X_train_ref = preprocess.load_and_preprocess_ref_images(df_train['ref'], ref_desired_size)
X_test_ref = preprocess.load_and_preprocess_ref_images(df_test['ref'], ref_desired_size)
#X_train_eval = preprocess.load_and_preprocess_eval_images(df_train['eval'], eval_desired_size)

Y_train=df_train['class']
Y_test=df_test['class']

#model
print(X_train_ref.shape)
#print(X_train2.shape)
model_gamma.model1(X_train_ref, Y_train, X_test_ref, Y_test)


