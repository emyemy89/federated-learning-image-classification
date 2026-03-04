import torch
import copy
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

import sys
import os
sys.path.append(os.path.abspath("../.."))

from src.client_training import train_local, evaluate_model, aggregate
from src.CNN_implementation import CNN
from src.plotting import plot_loss_overlay, plot_acc_overlay  # the new file


# ─── Shared Data Setup ────────────────────────────────────────────────────────

transform = transforms.ToTensor()

full_mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_size = int(0.7 * len(full_mnist_trainset))
val_size   = len(full_mnist_trainset) - train_size
mnist_trainset, mnist_valset = random_split(full_mnist_trainset, [train_size, val_size])
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

val_loader  = DataLoader(mnist_valset,  batch_size=64, shuffle=False)
test_loader = DataLoader(mnist_testset, batch_size=64, shuffle=False)


# ─── 1. Centralized Training ─────────────────────────────────────────────────

print("=" * 40)
print("CENTRALIZED TRAINING")
print("=" * 40)

number_of_epochs = 3
train_loader = DataLoader(mnist_trainset, batch_size=64, shuffle=True)

central_model = CNN()
_, central_losses, central_val_accs = train_local(
    central_model, train_loader, val_loader, number_of_epochs, lr=0.01
)

central_test_acc = evaluate_model(test_loader, central_model, True)
print(f"Centralized Test Accuracy: {central_test_acc:.2f}%")


# ─── 2. Federated Training (FedAvg) ──────────────────────────────────────────

print("\n" + "=" * 40)
print("FEDERATED TRAINING (FedAvg)")
print("=" * 40)

number_of_clients = 5
number_of_rounds  = 3
epochs            = 1

trainset_size        = len(mnist_trainset)
client_trainset_size = trainset_size // number_of_clients
client_trainsets     = random_split(mnist_trainset, [client_trainset_size] * number_of_clients)
client_training_loaders = [DataLoader(ds, batch_size=64, shuffle=True) for ds in client_trainsets]

global_model       = CNN()
fed_losses         = []
fed_val_accs       = []

for round_num in range(number_of_rounds):
    print(f"\nRound {round_num + 1}:")

    client_state_dicts = []
    num_samples        = []
    round_losses       = []

    for client in range(number_of_clients):
        print(f"  Client {client + 1}:")
        local_model = copy.deepcopy(global_model)
        weights, losses, _ = train_local(
            model=local_model,
            train_loader=client_training_loaders[client],
            number_of_epochs=epochs,
            lr=0.1
        )
        client_state_dicts.append(weights)
        num_samples.append(len(client_training_loaders[client].dataset))
        round_losses.append(losses[-1])

    round_loss = sum(round_losses) / len(round_losses)
    fed_losses.append(round_loss)

    global_model = aggregate(num_samples, client_state_dicts, global_model)

    print(f"  Avg Loss: {round_loss:.4f}")
    round_val_acc = evaluate_model(val_loader, global_model, True)
    fed_val_accs.append(round_val_acc)

fed_test_acc = evaluate_model(test_loader, global_model, True)
print(f"\nFederated Test Accuracy: {fed_test_acc:.2f}%")


# ─── 3. Overlay Plots ────────────────────────────────────────────────────────

plot_loss_overlay(central_losses, fed_losses)
plot_acc_overlay(central_val_accs, fed_val_accs)