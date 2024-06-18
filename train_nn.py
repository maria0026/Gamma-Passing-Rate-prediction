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
import numpy as np


def train(train_dataset, train_loader, model, criterion, optimizer, num_epochs, classes):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_loss = []
        for inputs, labels in train_loader:
            optimizer.zero_grad()

            if not classes:
                labels=labels.view(-1, 1)
            print(inputs.shape)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_dataset)
        all_loss.append(epoch_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    all_loss=np.array(all_loss)
    #save to txt
    np.savetxt('loss.txt', all_loss)
    #Pobranie wyuczonych wag
    trained_weights = model.state_dict()
    #Zapisanie wag do pliku
    torch.save(trained_weights, 'trained_weights.pth')



