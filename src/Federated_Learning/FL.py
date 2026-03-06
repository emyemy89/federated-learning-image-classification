# %%
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import copy
from torch.utils.data import DataLoader
from torch.utils.data import random_split # use for data distribution for clients
from collections import Counter

import sys
import os

from client_training import run_fl, train_local, evaluate_model, aggregate, create_IID_client_loaders, create_non_IID_client_loaders
from src.CNN_implementation import CNN
sys.path.append(os.path.abspath("../.."))



from src.plotting import plot_loss, plot_acc

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
NUMBER_OF_CLIENTS = 5
# %%
# Data Loaders
client_training_loaders = create_non_IID_client_loaders(number_of_clients = NUMBER_OF_CLIENTS, mnist_trainset = mnist_trainset)

'''DEBUG_CLIENT_DISTRIBUTION = True
if DEBUG_CLIENT_DISTRIBUTION:
    print("\nSampled label distribution per client (first few batches):")
    for client_id, loader in enumerate(client_training_loaders):
        label_counts = Counter()
        for batch_idx, (_, labels) in enumerate(loader):
            label_counts.update(labels.tolist())
            if batch_idx >= 9:  # sample first 10 batches only
                break
        digits = sorted(label_counts.keys())
        print(
            f"Client {client_id}: digits={digits} "
            f"counts={{" + ", ".join(f"{d}:{label_counts[d]}" for d in digits) + "}}"
        )
'''

val_loader = DataLoader(mnist_valset, batch_size = 64, shuffle = False)
test_loader = DataLoader(mnist_testset, batch_size=64, shuffle=False)

# %%
# Now we need to aggregate the results with FedAvg
run_fl(number_of_rounds = 3,
       number_of_clients = NUMBER_OF_CLIENTS,
       epochs = 3,
       global_model = global_model,
       client_training_loaders = client_training_loaders,
       val_loader = val_loader,
       test_loader = test_loader)




