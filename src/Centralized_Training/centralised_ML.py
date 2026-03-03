# %%
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import sys
import os
sys.path.append(os.path.abspath("../.."))

from src.client_training import train_local, evaluate_model
from src.CNN_implementation import CNN
# %%
# We will need to convert image to tensors
# here we define a transform
transform = transforms.ToTensor()
# %%
# load the dataset
# now each image mnist_trainset[i][0] is a tensor of shape [1, 28, 28] (1 channel, 28×28 pixels)
full_mnist_trainset = datasets.MNIST(root = './data', train = True, download = True, transform = transform)
train_size = int(0.7 * len(full_mnist_trainset))
val_size = len(full_mnist_trainset) - train_size
mnist_trainset, mnist_valset = random_split(full_mnist_trainset, [train_size, val_size])

mnist_testset = datasets.MNIST(root = './data', train = False, download = True, transform = transform)


image, label = mnist_trainset[0]

print(type(image))
print(image.shape)
print(label)
# %%
#Create CNN

x = image.unsqueeze(0)
model = CNN()
out = model(x)
print(out.shape)


# %%
#Train
number_of_epochs = 3

train_loader = DataLoader(mnist_trainset, batch_size = 64, shuffle = True)
val_loader = DataLoader(mnist_valset, batch_size = 64, shuffle = False)

# do training
train_local( model, train_loader, val_loader, number_of_epochs, 0.01)

# Test the accuracy on test data
test_loader = DataLoader(mnist_testset, batch_size = 64, shuffle = False)# loading

test_accuracy = evaluate_model(test_loader, model)
print(f"Test Accuracy: {test_accuracy:.2f}%")
