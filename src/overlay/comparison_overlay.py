# %%
import sys
import os
sys.path.append(os.path.abspath("../.."))
print(sys.executable)

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from src.client_training import run_fl, run_centralised_ml
from src.data_loading import create_dirichlet_client_loaders, debug_non_iid_split
from src.model_implementation import CNN, MLP
from src.plotting import (
    plot_loss_overlay,
    plot_acc_overlay,
    plot_client_label_distribution,
)


# %%
# Data Loading
transform = transforms.ToTensor()

full_mnist_trainset = datasets.MNIST(root = './data', train = True, download = True, transform = transform)
train_size = int(0.7 * len(full_mnist_trainset))
val_size = len(full_mnist_trainset) - train_size

mnist_trainset, mnist_valset = random_split(full_mnist_trainset, [train_size, val_size])
mnist_testset = datasets.MNIST(root = './data', train = False, download = True, transform = transform)
# %%
# the model: CNN or MLP
gl_model = CNN()
model = CNN()

NUMBER_OF_CLIENTS = 5
NUMBER_OF_EPOCHS = 3
NUMBER_OF_ROUNDS = 10
DIRICHLET_ALPHA = 1
BATCH_SIZE = 64
LR = 0.03
# %%
# Data Loaders
client_training_loaders = create_dirichlet_client_loaders(number_of_clients = NUMBER_OF_CLIENTS,
                                                          mnist_trainset = mnist_trainset,
                                                          alpha = DIRICHLET_ALPHA)
train_loader = DataLoader(mnist_trainset, batch_size = BATCH_SIZE, shuffle = True)

# this will show what digits each client has (console)
# debug_non_iid_split(client_training_loaders)

val_loader = DataLoader(mnist_valset, batch_size = BATCH_SIZE, shuffle = False)
test_loader = DataLoader(mnist_testset, batch_size = BATCH_SIZE, shuffle=False)

# %%
central_losses, central_val_accs  =  run_centralised_ml(number_of_epochs = NUMBER_OF_ROUNDS,
                                                      train_loader = train_loader,
                                                      val_loader = val_loader,
                                                      test_loader = test_loader,
                                                      model = model,
                                                      lr = 0.01)

# Now we need to aggregate the results with FedAvg
fed_losses, fed_val_accs        =       run_fl(number_of_rounds = NUMBER_OF_ROUNDS,
                                              number_of_clients = NUMBER_OF_CLIENTS,
                                              epochs = NUMBER_OF_EPOCHS,
                                              global_model = gl_model,
                                              client_training_loaders = client_training_loaders,
                                              val_loader = val_loader,
                                              test_loader = test_loader,
                                              lr = 0.05,
                                               participation_rate = 1)

# PLOTS
plot_loss_overlay(central_losses, fed_losses) # loss comparison between ML and FL
plot_acc_overlay(central_val_accs, fed_val_accs) # accuracy comparison between ML and FL
plot_client_label_distribution(client_training_loaders) # create a heatmap-style overview per client