# %%
import sys
import os
sys.path.append(os.path.abspath("../../.."))

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from training_paradigms.lib.utils_client_training import run_centralised_ml
from models.model_implementation import CNN

# %%
# load the dataset
# now each image mnist_trainset[i][0] is a tensor of shape [1, 28, 28] (1 channel, 28×28 pixels)
transform = transforms.ToTensor()
full_mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_size = int(0.7 * len(full_mnist_trainset))
val_size = len(full_mnist_trainset) - train_size

mnist_trainset, mnist_valset = random_split(full_mnist_trainset, [train_size, val_size])
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# %%
# Create CNN
model = CNN()

# data loaders
train_loader = DataLoader(mnist_trainset, batch_size = 64, shuffle = True)
val_loader = DataLoader(mnist_valset, batch_size = 64, shuffle = False)
test_loader = DataLoader(mnist_testset, batch_size=64, shuffle=False)

# function that does it all
run_centralised_ml(number_of_epochs = 3,
                   train_loader = train_loader,
                   val_loader = val_loader,
                   test_loader = test_loader,
                   model = model)