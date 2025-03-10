import os
import random
import h5py
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.optimizer import Optimizer
import torch.utils.data as data
import matplotlib.pyplot as plt
from torchvision import transforms, datasets, models
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from time import time
from torchsummary import summary
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from time import time
from torchvision.datasets import ImageFolder
from model.function import *
from model.module import *
import argparse
from thop import profile

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

def predict():
    set_seed()
    premodel1 = []
    premodel2 = []
    premodel1 = UNet(3, 64)
    premodel2 = UNet(3, 64)
    
    num_class = 5
    batch_size = 8
    drop = 0.1
    dense_1 = 128
    dense_2 = 64

    doppler, ranges, premodel1, premodel2 = load_weight(premodel1, premodel2)
    model = Classificate(premodel1, premodel2, num_class, dense_1, dense_2, drop)
    model_path = './model/weights/model.pth'
    model.load_state_dict(torch.load(model_path))
    premodel1 = premodel1.to(device)
    premodel2 = premodel2.to(device)
    model = model.to(device)
    model.eval()

    doppler = doppler.to(device)
    ranges = ranges.to(device)

    with torch.no_grad():
        outputs = model(doppler, ranges)
        _, predicted = torch.max(outputs, 1)
        
        # print(str(predicted.cpu().numpy()[0]))
        return int(predicted.cpu().numpy()[0])

if __name__ == "__main__":
    set_seed()
    predict()