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

#sorting the classes into different folders
source_path='data/reference_nn'
desired_column='ref'
#preprocess.split_folder(df=df,source_path=source_path, desired_column=desired_column)

#processing dcm files to jpg for all folders
#preprocess.preprocess_folder(source_path)

#2 classes
csv_file='dataframes/data_for_nn5_final.csv'

#Create data instance

root='/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/reference'
root='/home/mariaw/gpr/data/reference'
classes= False
train_dataset, test_dataset, train_loader, test_loader=preprocess.initialize_data(csv_file, root, classes= classes)

#train the model
num_classes=2
model = model_gamma.CustomModel_2()
#Define loss function and optimizer
criterion = nn.BCELoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs=10

train_nn.train(train_dataset, train_loader, model, criterion, optimizer, num_epochs, classes=classes)

#evaluate the model
eval_nn.eval(test_loader, model, num_classes)

#loading weights
loaded_weights = torch.load('trained_weights_2_klasy.pth')
model.load_state_dict(loaded_weights)

#10 classes
root='/home/mariaw/gpr/data/reference_nn'
root='/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/reference_nn'
classes=True
train_dataset, test_dataset, train_loader, test_loader=preprocess.initialize_data(csv_file, root, classes=classes)
num_classes=10
model = model_gamma.CustomModel(num_classes=num_classes)
'''
#Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs=10

train_nn.train(train_dataset, train_loader, model, criterion, optimizer, num_epochs, classes=classes)



#evaluate the model
eval_nn.eval(test_loader, model, num_classes)

#loading weights
loaded_weights = torch.load('trained_weights.pth')
model.load_state_dict(loaded_weights)
'''