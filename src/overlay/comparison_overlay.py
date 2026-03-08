# %%
import sys
import os
sys.path.append(os.path.abspath("../.."))
print(sys.executable)

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split # use for data distribution for clients

from client_training import run_fl, run_centralised_ml
from data_loading import create_dirichlet_client_loaders, debug_non_iid_split
from src.CNN_implementation import CNN
from src.plotting import plot_loss_overlay, plot_acc_overlay  # the new file


# %%
# Data Loading
transform = transforms.ToTensor()

full_mnist_trainset = datasets.MNIST(root = './data', train = True, download = True, transform = transform)
train_size = int(0.7 * len(full_mnist_trainset))
val_size = len(full_mnist_trainset) - train_size

mnist_trainset, mnist_valset = random_split(full_mnist_trainset, [train_size, val_size])
mnist_testset = datasets.MNIST(root = './data', train = False, download = True, transform = transform)
# %%
# CNN
# a CNN with 2 layers: Layer1-> Relu->Pool->Layer2->...->Flatten->Fully connected->Output
gl_model = CNN()
model = CNN()
NUMBER_OF_CLIENTS = 5
# %%
# Data Loaders
client_training_loaders = create_dirichlet_client_loaders(number_of_clients = NUMBER_OF_CLIENTS,
                                                          mnist_trainset = mnist_trainset,
                                                          alpha = 0.5)
train_loader = DataLoader(mnist_trainset, batch_size = 64, shuffle = True)
###
# this will show what digits each client has
debug_non_iid_split(client_training_loaders)

val_loader = DataLoader(mnist_valset, batch_size = 64, shuffle = False)
test_loader = DataLoader(mnist_testset, batch_size=64, shuffle=False)

# %%
central_losses, central_val_accs = run_centralised_ml(number_of_epochs = 3,
                   train_loader = train_loader,
                   val_loader = val_loader,
                   test_loader = test_loader,
                   model = model)
# Now we need to aggregate the results with FedAvg
fed_losses, fed_val_accs = run_fl(number_of_rounds = 3,
       number_of_clients = NUMBER_OF_CLIENTS,
       epochs = 3,
       global_model = gl_model,
       client_training_loaders = client_training_loaders,
       val_loader = val_loader,
       test_loader = test_loader)

plot_loss_overlay(central_losses, fed_losses)
plot_acc_overlay(central_val_accs, fed_val_accs)