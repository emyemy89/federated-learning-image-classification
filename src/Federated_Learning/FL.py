# %%
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import copy
from torch.utils.data import DataLoader
from torch.utils.data import random_split # use for data distribution for clients


import sys
import os
sys.path.append(os.path.abspath("../.."))

from src.client_training import train_local,  evaluate_model, aggregate
from src.CNN_implementation import CNN
print(sys.executable)
# %%
# Data Loading
transform = transforms.ToTensor()

full_mnist_trainset = datasets.MNIST(root = './data', train = True, download = True, transform = transform)
train_size = int(0.7 * len(full_mnist_trainset))
val_size = len(full_mnist_trainset) - train_size
mnist_trainset, mnist_valset = random_split(full_mnist_trainset, [train_size, val_size])

mnist_testset = datasets.MNIST(root = './data', train = False, download = True, transform = transform)
image, label = mnist_trainset[0]
# %%
# CNN
# a CNN with 2 layers: Layer1-> Relu->Pool->Layer2->...->Flatten->Fully connected->Output

global_model = CNN()
out = global_model(image.unsqueeze(0))
print(out.shape)
# %%
# Data Loaders
# Client abstraction, split training dataset into clients
number_of_clients = 5

trainset_size = len(mnist_trainset)
client_trainset_size = trainset_size // number_of_clients # we use // instead of / to remove the fractional part
client_trainsets = random_split(mnist_trainset, [client_trainset_size]*number_of_clients) # a list with: client_datasets[0], client_datasets[1], ...[5]

client_training_loaders = []
for dataset in client_trainsets:
    train_loader = DataLoader(dataset, batch_size = 64, shuffle = True)
    client_training_loaders.append(train_loader)
    
val_loader = DataLoader(mnist_valset, batch_size = 64, shuffle = False)
test_loader = DataLoader(mnist_testset, batch_size=64, shuffle=False)

# print(len(client_loaders))
# print(len(client_loaders[0]))
# images, labels = next(iter(client_loaders[0]))
# print(images.shape)
# %%
# Now we need to aggregate the results with FedAvg
number_of_rounds = 3
for round in range(number_of_rounds):
    print("---------------------------------")
    print(f"Round {round+1}:\n")
    
    client_state_dictionary = []
    number_of_samples = []
    # loop over all clients
    for client in range(number_of_clients):
        print(f"Client{client+1}:")
        local_model = copy.deepcopy(global_model) # copy global model locally
        weights = train_local(model = local_model, train_loader = client_training_loaders[client], number_of_epochs = 1, lr = 0.1) # train it
        client_state_dictionary.append(weights) # store the weights
        number_of_samples.append(len(client_training_loaders[client].dataset)) # store number of samples
        
    # aggregate
    global_model = aggregate(number_of_samples, client_state_dictionary, global_model)
    print("Validation:")
    evaluate_model(val_loader, global_model)
# compute accuracy
evaluate_model(test_loader, global_model)


