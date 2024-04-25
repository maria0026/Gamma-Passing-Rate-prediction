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


#prepraring df
data_folder='/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/data_final'
extension = 'dcm'
first_keyword = 'Predicted'
second_keyword='Portal'

reference_folder = '/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/reference'
evaluation_folder = '/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/evaluation'
txt_folder = '/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/txt'

#sort data from data folder into reference and evaluation folders
prepare_df.data_sorting(data_folder, extension, first_keyword, second_keyword)
files_without_pair=prepare_df.find_dcm_without_pair(reference_folder, evaluation_folder)
print("Dcm without pair", len(files_without_pair))
#there are some files (825) without a pair, we will not take them into account

#prepare df with reference, evaluation and txt files
df=prepare_df.create_df(reference_folder, evaluation_folder, txt_folder)
df[['ref', 'eval', 'txt']].to_csv('dataframes/ref_eval3.csv', sep=';', index=False)
df=pd.read_csv('dataframes/ref_eval3.csv', sep=';')
print(df)

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
df['class'] = df['gamma_txt'].apply(preprocess.map_to_class)

#sorting the classes into different folders
source_path='data/reference_nn'
desired_column='ref'
#preprocess.split_folder(df=df,source_path=source_path, desired_column=desired_column)

#processing dcm files to jpg for all folders
#preprocess.preprocess_folder(source_path):


#Define transformations
transform = transforms.Compose([
    #transforms.Resize((224, 224)),  # Resize images to a fixed size
    transforms.ToTensor(),           # Convert images to PyTorch tensors
])
def loader(path):
    i=0
    image, size= normalize.normalize_ref_image(path)
    image=image.astype('float32')

    print(image.dtype)
    #print(i)
    if size==(1024, 1024):
        return image
    
def is_valid_file(x):
    return x.endswith('.dcm')

#Create ImageFolder instance
root='/home/marysia/Documents/GitHub/Gamma-Passing-Rate-prediction/data/reference_nn'
dataset = ImageFolder(root=root,transform=transform,loader=loader, is_valid_file=is_valid_file)

#Define the size of the training set
train_size = int(0.8 * len(dataset))

#Randomly split indices for training and testing sets
indices = list(range(len(dataset)))
random.shuffle(indices)
train_indices, test_indices = indices[:train_size], indices[train_size:]

#Create subset datasets and data loaders for training and testing
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)
print(train_dataset)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

#train nn 
#Instantiate the model
num_classes=10
model = model_gamma.CustomModel()
#Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


#Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        #inputs=inputs.double()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
'''
#Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Test Accuracy: {accuracy:.4f}')
'''