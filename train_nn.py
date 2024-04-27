import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms
import random
import model_gamma
import normalize
from skimage.transform import resize



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
    else:
        image = resize(image, (1024, 1024))
        return image

    
def is_valid_file(x):
    return x.endswith('.dcm')

def initialize_data(root):
    #Create ImageFolder instance

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


    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    return train_dataset, test_dataset, train_loader, test_loader


def train(train_dataset, train_loader, model, criterion, optimizer, num_epochs):
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
