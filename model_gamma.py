
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


#X_train, Y_train, X_test, Y_test

from torch.utils.data import Dataset
from PIL import Image

#first nn- give only reference and gamma passing rate as an input; second nn- give reference and evaluation and gamma passing rate as an input
class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.activation = F.relu

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2)

        self.adaptivepool = nn.AdaptiveAvgPool2d((1,1))

        self.linear1 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool1(self.activation(self.bn1(self.conv1(x))))
        x = self.pool2(self.activation(self.bn2(self.conv2(x))))
        x = self.pool3(self.activation(self.bn3(self.conv3(x))))
        x = self.pool4(self.activation(self.bn4(self.conv4(x))))
        x = self.adaptivepool(x)
        x = torch.flatten(x, 1) 
        x = self.linear1(x)
        return x




