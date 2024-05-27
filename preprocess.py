import normalize
import numpy as np
import torch
import os
import shutil
import pydicom
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision import transforms
import random 
from skimage.transform import resize
from torch.utils.data import DataLoader, Dataset, Subset
import pandas as pd

def map_to_class(gamma_value):
    if gamma_value < 95:
        return 0
    elif 95 <= gamma_value < 96:
        return 1
    elif 96 <= gamma_value < 96.5:
        return 2
    elif 96.5 <= gamma_value < 97:
        return 3
    elif 97 <= gamma_value < 98:
        return 4
    elif 98 <= gamma_value < 99:
        return 5
    elif 99 <= gamma_value < 99.5:
        return 6
    elif 99.5 <= gamma_value < 99.8:
        return 7
    elif 99.8 <= gamma_value < 99.9:
        return 8
    elif 99.9 <= gamma_value <= 100:
        return 9
    else:
        return None
    
def split_folder(df,source_path,desired_column):

    for index, row in df.iterrows():
        file_path = row[desired_column]
        print(file_path)
        file_path=file_path.replace("  ", " ")
        filename=os.path.basename(file_path)
        destination_folder = os.path.join(source_path, str(row['class']))
        destination_path = os.path.join(destination_folder, filename)

        os.makedirs(destination_folder, exist_ok=True)

        #Check if the source file exists before attempting to move it
        if os.path.exists(file_path):
            #Move the file
            shutil.move(file_path, destination_path)
            print(f"Moved {file_path} to {destination_path}")
        else:
            print(f"Source file {file_path} does not exist")


#Define transformations

#transform for radiation distribution
transform = transforms.Compose([
    #transforms.Lambda(transform_sample),
    # Add more transformations if needed
    transforms.ToTensor(),  
])

#Transsform for GPR value
def transform_gpr(gpr):
    #Convert gpr to float32 explicitly
    gpr = np.float32(gpr)
    gpr=torch.tensor(gpr, dtype=torch.float32) 
    return gpr

target_transform = transforms.Compose([
    transforms.Lambda(lambda gpr: transform_gpr(gpr))
])

def loader(path):
    image, size= normalize.normalize_ref_image(path)
    image=image.astype('float32')
    print("Image shape: ", image.shape)
    if size==(1024, 1024):
        return image
    else:
        image = resize(image, (1024, 1024))
        return image

    
def is_valid_file(x):
    return x.endswith('.dcm')

class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=transform, target_transform=target_transform, loader=loader):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            loader (callable, optional): Optional function to load an image given its path.
        """
        self.landmarks_frame = pd.read_csv(csv_file, sep=';')
        self.landmarks_frame = self.landmarks_frame.dropna()  # Drop rows with 'None' values
        self.landmarks_frame = self.landmarks_frame[self.landmarks_frame['pass_ratio'] > 90]  # Filter GPR > 90
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #0 index is reference distribution
        img_name = self.landmarks_frame.iloc[idx, 0]
        
        #Load image
        image = self.loader(img_name)

        #Load GPR from 3rd column
        gpr = self.landmarks_frame.iloc[idx, 2]

        if self.transform:
            image=self.transform(image)

        if self.target_transform:
            gpr=self.target_transform(gpr)
            gpr=gpr/100

        return image,gpr.float()
    
def initialize_data(csv_file, root, classes):
    #Create ImageFolder instance
    if not classes:
        dataset = CustomDataset(csv_file=csv_file, root_dir=root, transform=transform, target_transform=target_transform, loader=loader)
    else:
        dataset = ImageFolder(root=root,transform=transform,loader=loader, is_valid_file=is_valid_file)

    #Define the size of the training set
    train_size = int(0.8 * len(dataset))
    val_size=int(0.1*len(dataset))
    test_size=int(0.1*len(dataset))

    #Randomly split indices for training and testing sets
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    train_indices, test_indices = indices[:train_size], indices[train_size:]

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    #Create subset datasets and data loaders for training and testing
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


