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
import drzewo


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
df = pd.read_csv('dataframes/data_for_nn5_torch.csv', sep=',')
print(df)
df = df.dropna()
df=df[df['pass_ratio']>90]
print(df)
#df['class'] = df['gamma_txt'].apply(preprocess.map_to_class)

#sorting the classes into different folders
source_path='data/reference_nn'
desired_column='ref'
#preprocess.split_folder(df=df,source_path=source_path, desired_column=desired_column)

#processing dcm files to jpg for all folders
#preprocess.preprocess_folder(source_path)

#Create data instance
root='/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/reference_nn'
train_dataset, test_dataset, train_loader, test_loader=train_nn.initialize_data(root)


#train the model
num_classes=10
model = model_gamma.CustomModel(num_classes=num_classes)
#Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs=10
train_nn.train(train_dataset, train_loader, model, criterion, optimizer, num_epochs)

#loading weights
loaded_weights = torch.load('trained_weights_2_klasy.pth')
model.load_state_dict(loaded_weights)

#ewaluate the model
eval_nn.eval(test_loader, model)
