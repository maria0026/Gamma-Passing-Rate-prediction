import os
import pandas as pd
import prepare_df
import gamma
import preprocess 
from sklearn.model_selection import train_test_split
import model_gamma
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
import random
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import pydicom
import normalize
from skimage.transform import resize
import train_nn
import eval_nn
import valid
import numpy as np
import matplotlib.pyplot as plt

#prepraring df
data_folder='/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/data_final'
extension = 'dcm'
first_keyword = 'Predicted'
second_keyword='Portal'

reference_folder = '/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/reference'
evaluation_folder = '/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/evaluation'
txt_folder = '/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/txt'

#sort data from data folder into reference and evaluation folders
#prepare_df.data_sorting(data_folder, extension, first_keyword, second_keyword)
#files_without_pair=prepare_df.find_dcm_without_pair(reference_folder, evaluation_folder)
#print("Dcm without pair", len(files_without_pair))
#there are some files (825) without a pair, we will not take them into account

#prepare df with reference, evaluation and txt files
#df=prepare_df.create_df(reference_folder, evaluation_folder, txt_folder)
#df[['ref', 'eval', 'txt']].to_csv('dataframes/ref_eval3.csv', sep=';', index=False)
#df=pd.read_csv('dataframes/ref_eval3.csv', sep=';')
#print(df)

#calculate gamma passing rate for df
#df = pd.read_csv('ref_eval2.csv', sep=';')
#print(df)
dose_percent_threshold=2
distance_mm_threshold=2
lower_percent_dose_cutoff=5
#df=gamma.calculate_gamma_for_df(df, dose_percent_threshold=dose_percent_threshold, distance_mm_threshold=distance_mm_threshold, lower_percent_dose_cutoff=lower_percent_dose_cutoff, draw=False)
#df[['pass_ratio','gamma_txt']].to_csv('gamma_results5.csv', sep='\t', index=False)


#load and preprocess images
#df = pd.read_csv('dataframes/data_for_nn5_naprawione.csv', sep=';')
#df=prepare_df.calculate_size(df)
#df.to_csv('dataframes/data_for_nn5_final.csv', sep=';', index=False)

df=pd.read_csv('dataframes/data_for_nn5_final.csv', sep=';')
#df["ref_size_condition"]=df['ref_size']=="(1024, 1024)"
#df["eval_size_condition"]=df['eval_size']=="(1190, 1190)"
#df.to_csv('dataframes/data_for_nn5_final.csv', sep=';', index=False)
print(df)
df = df.dropna()
df=df[df['pass_ratio']>90]
print(df)
#print((df['ref_size']!="(1024, 1024)").sum())

#df['class'] = df['gamma_txt'].apply(preprocess.map_to_class)
df=df.dropna()
'''
#remove rows containing False in ref_size_condition
df=df[df['ref_size_condition']==True]
df=df[df['eval_size_condition']==True]
#save df to csv
df.to_csv('dataframes/data_for_nn5_final_size.csv', sep=';', index=False)
'''
#zrobic to sortowanie dla nowego df
'''
df=pd.read_csv('dataframes/data_for_nn5_final_size.csv', sep=';')
df['class'] = df['gamma_txt'].apply(preprocess.map_to_class)
#sorting the classes into different folders
source_path='data/reference_nn_2'
desired_column='ref'
preprocess.split_folder(df=df,source_path=source_path, desired_column=desired_column)
'''
#processing dcm files to jpg for all folders
#preprocess.preprocess_folder(source_path)


#Create data instance
#10 classes
csv_file='dataframes/data_for_nn5_final_size.csv'
root='/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/reference_nn'
root='/home/mariaw/gpr/data/reference_nn_2'
classes=True
num_classes=10
model = model_gamma.CustomModel(num_classes=num_classes)

#Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs=10
#cross validation

number_of_folds=1
f1_scores=[]
specificities=[]
gain=[]
'''
for i in range(number_of_folds):
    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader=preprocess.initialize_data(csv_file, root, classes=classes)
    
    #train the model
    train_nn.train(train_dataset, train_loader, model, criterion, optimizer, num_epochs, classes=classes)

    #validate the model
    #val_nn.validate(val_loader, model, num_classes)

    #evaluate the model
    eval_nn.eval(test_loader, model, num_classes)

    #loading weights
    #loaded_weights = torch.load('trained_weights.pth')
    #model.load_state_dict(loaded_weights)
'''

csv_file='dataframes/probabilities_1.csv'
df=pd.read_csv(csv_file, sep=',')
csv_file='dataframes/probabilities_2.csv'
df=pd.read_csv(csv_file, sep=',')
classes_cutoff=3
min_class=7
max_class=9

#split df 5 times using monte carlo ale calculate threshold for each split
val_size=0.5
thresholds=[]
FPs=[]
gains=[]
classes_cutoffs=[2, 3, 4, 5, 6, 7]

#classes_cutoffs=[2,3,4]
min_class=7
#max_class=min_class
max_class=9
nr_of_splits=20

csv_file='dataframes/probabilities_1_3.csv'
df=pd.read_csv(csv_file, sep=',')
#valid.calculate_threshold_and_find_gain(df, val_size, classes_cutoffs, min_class, max_class, nr_of_splits)
csv_file='dataframes/probabilities_2_3.csv'
df=pd.read_csv(csv_file, sep=',')
#valid.calculate_threshold_and_find_gain(df, val_size, classes_cutoffs, min_class, max_class, nr_of_splits)
csv_file='dataframes/probabilities_3.csv'
df=pd.read_csv(csv_file, sep=',')
#valid.calculate_threshold_and_find_gain(df, val_size, classes_cutoffs, min_class, max_class, nr_of_splits)
csv_file='dataframes/probabilities_4.csv'
df=pd.read_csv(csv_file, sep=',')
#valid.calculate_threshold_and_find_gain(df, val_size, classes_cutoffs, min_class, max_class, nr_of_splits)


csv_file='dataframes/probabilities_1_2.csv'
df=pd.read_csv(csv_file, sep=',')
valid.calculate_threshold_and_find_gain(df, val_size, classes_cutoffs, min_class, max_class, nr_of_splits)
csv_file='dataframes/probabilities_2_2.csv'
df=pd.read_csv(csv_file, sep=',')
valid.calculate_threshold_and_find_gain(df, val_size, classes_cutoffs, min_class, max_class, nr_of_splits)
csv_file='dataframes/probabilities_3_2.csv'
df=pd.read_csv(csv_file, sep=',')
valid.calculate_threshold_and_find_gain(df, val_size, classes_cutoffs, min_class, max_class, nr_of_splits)
csv_file='dataframes/probabilities_4_2.csv'
df=pd.read_csv(csv_file, sep=',')
valid.calculate_threshold_and_find_gain(df, val_size, classes_cutoffs, min_class, max_class, nr_of_splits)
csv_file='dataframes/probabilities_5_2.csv'
df=pd.read_csv(csv_file, sep=',')
valid.calculate_threshold_and_find_gain(df, val_size, classes_cutoffs, min_class, max_class, nr_of_splits)
'''
classes_cutoffs=[1,2,3, 4, 5, 6, 7]
min_class=7
#max_class=min_class
max_class=9
csv_file='dataframes/probabilities_1.csv'
df=pd.read_csv(csv_file, sep=',')
valid.calculate_threshold_and_find_gain_class(df, val_size, classes_cutoffs, min_class, max_class)

csv_file='dataframes/probabilities_2.csv'
df=pd.read_csv(csv_file, sep=',')
valid.calculate_threshold_and_find_gain_class(df, val_size, classes_cutoffs, min_class, max_class)
csv_file='dataframes/probabilities_3.csv'
df=pd.read_csv(csv_file, sep=',')
valid.calculate_threshold_and_find_gain_class(df, val_size, classes_cutoffs, min_class, max_class)
csv_file='dataframes/probabilities_4.csv'
df=pd.read_csv(csv_file, sep=',')
valid.calculate_threshold_and_find_gain_class(df, val_size, classes_cutoffs, min_class, max_class)
#w klasycznym podejściu nie jesteśmy w stanie odzielić klas tak aby przy jakimś progu móc uznać że FP=0
'''