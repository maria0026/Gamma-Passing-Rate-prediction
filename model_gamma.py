
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import softmax as fn_softmax
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from torchvision.models import resnet18
import torch.nn as nn
import torch.nn.functional as F
import normalize
from torch.utils.data import Dataset
from PIL import Image

#first nn- give only reference and gamma passing rate as an input; second nn- give reference and evaluation and gamma passing rate as an input

class CustomModel(nn.Module):
    def __init__(self, num_classes):
        super(CustomModel, self).__init__()
        self.resnet = resnet18()
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


class CustomModel_2(nn.Module):
    def __init__(self):
        super(CustomModel_2, self).__init__()
        self.resnet = resnet18()
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)

    def forward(self, x):
        x = self.resnet(x)
        x = torch.sigmoid(x)  # Funkcja sigmoidalna dla wartości pomiędzy 0 a 1
        return x

    
