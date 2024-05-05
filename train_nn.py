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



def train(train_dataset, train_loader, model, criterion, optimizer, num_epochs, classes):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            #print(labels.shape)
            if not classes:
                labels=labels.view(-1, 1)
            #print(labels.shape)
            #inputs=inputs.double()
            #print(type(inputs), type(labels))
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    
    #Pobranie wyuczonych wag
    trained_weights = model.state_dict()
    #Zapisanie wag do pliku
    torch.save(trained_weights, 'trained_weights_2_klasy.pth')



