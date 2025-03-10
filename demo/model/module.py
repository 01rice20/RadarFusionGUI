import os
import random
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, confusion_matrix
import torch.nn.functional as F
from time import time
from torchsummary import summary
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from time import time
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
from model.function import *
import math

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, out_c),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, out_c),
            nn.ReLU()
        )
  
    def forward(self, x):
        x = self.conv1(x)
        concate = x.clone()
        x = self.conv2(x)

        return concate, x

class Encoder(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        real_c = int(out_c/2)
        self.conv = DoubleConv(in_c, real_c)
        self.max_pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        concate, x = self.conv(x)
        x = torch.cat([concate, x], dim=1)
        x = self.max_pool(x)

        return x

class Decoder(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        real_c = int(out_c/2)
        self.conv = DoubleConv(in_c, real_c)
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        concate, x = self.conv(x)
        x = torch.cat([concate, x], dim=1)
        x = self.upsample(x)

        return x

class UNet(nn.Module):
    def __init__(self, channel, num_filter):
        super().__init__()
        self.encoder0 = DoubleConv(channel, num_filter)
        self.encoder1 = Encoder(num_filter, num_filter*2)
        self.encoder2 = Encoder(num_filter*2, num_filter*4)
        self.encoder3 = Encoder(num_filter*4, num_filter*8)
        self.encoder4 = Encoder(num_filter*8, num_filter*8)
        
        self.decoder1 = Decoder(num_filter*8, num_filter*8) 
        self.decoder2 = Decoder(num_filter*8, num_filter*4) 
        self.decoder3 = Decoder(num_filter*4, num_filter*2) 
        self.decoder4 = Decoder(num_filter*2, num_filter)
        self.decoder0 = nn.Conv2d(num_filter, channel, kernel_size=1)

    def forward(self, x):
        _, x = self.encoder0(x)
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        x = self.encoder4(x)

        x = self.decoder1(x)
        x = self.decoder2(x)
        x = self.decoder3(x)
        x = self.decoder4(x)
        x = self.decoder0(x)
        
        return x

class FineTune(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_c),
        )

        self.simatt = SimpleAttention(in_c)
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_c)
        )
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=2, stride=2)

        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        ones(self.rate1)
        ones(self.rate2)

    def forward(self, x):
        set_seed()
        concate = x
        concate = self.downsample(concate)
        
        # Add Learning Parameters
        simattx = self.conv1(x)
        simattx = self.simatt(simattx)
        x = self.conv(x)
        x = self.rate1*x + self.rate2*simattx

        return x + concate

class Classificate(nn.Module):
    def __init__(self, model1, model2, num_classes, dense_1, dense_2, drop):
        super().__init__()
        # dense1_inp 1000 for resatt, 3*128*128 for original model, 128*8*8 for convblock
        self.model1 = model1
        self.model2 = model2
        
        # New Method
        self.convblock1 = FineTune(6, 32)
        self.convblock2 = FineTune(32, 64)
        self.convblock3 = FineTune(64, 64)
        self.convblock4 = FineTune(64, 128)

        self.dense_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*8*8, dense_1),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(dense_1, dense_2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(dense_2, num_classes)
        )
        
    def forward(self, x1, x2):
        set_seed()
        x1 = self.model1(x1)
        x2 = self.model2(x2)

        # Concatentation
        x = torch.cat([x1, x2], dim=1)
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.dense_layer(x)
        
        return x

class SimpleAttention(nn.Module):

    def __init__(self, channels = None, e_lambda = 1e-4):
        super().__init__()

        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward (self, x):
        b, c, h, w = x.size()
        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activation(y)