# %%
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import sys
import os
sys.path.append(os.path.abspath("../.."))

from trainingGPU import train_model
from testingGPU import model_testing
from src.CNN_implementation import CNN
# %%
# We will need to convert image to tensors
# here we define a transform
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

transform = transforms.ToTensor()
# %%
# load the dataset
# now each image mnist_trainset[i][0] is a tensor of shape [1, 28, 28] (1 channel, 28×28 pixels)
mnist_trainset=datasets.MNIST(root = './data', train = True, download = True, transform = transform)
mnist_testset=datasets.MNIST(root = './data', train = False, download = True, transform = transform)
# %%
#len(mnist_trainset) len(mnist_testset)
# %%
image, label = mnist_trainset[0]

print(type(image))
print(image.shape)
print(label)
# %%
#Create CNN
x = image.unsqueeze(0).to(device)
model = CNN().to(device)
out = model(x)
print(out.shape)
# %%
# Training
train_loader = DataLoader(mnist_trainset, batch_size = 64, shuffle = True)
train_model(train_loader, model, device)

# Test the accuracy on test data
test_loader = DataLoader(mnist_testset, batch_size = 64, shuffle = False)# loading

model_testing(test_loader, model, device)

