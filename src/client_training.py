import torch
import copy
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader, Subset

from plotting import plot_loss, plot_acc


def train_local(model, train_loader, val_loader = None, number_of_epochs = 1, lr = 0.1):
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.SGD(model.parameters(), lr = lr)
    model.train()
    losses, val_accuracies = [], []
    for epoch in range (number_of_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            # forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # backward pass
            optimiser.zero_grad()
            loss.backward()
            optimiser.step() 
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        losses.append(avg_loss) 
        if val_loader is not None:
            val_accuracy = evaluate_model(val_loader, model, False)
            val_accuracies.append(val_accuracy)
            print(f"Epoch {epoch+1}/{number_of_epochs}, Loss: {avg_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        else:
            print(f"Epoch {epoch+1}/{number_of_epochs}, Loss: {avg_loss:.4f}")
    return model.state_dict(), losses, val_accuracies # return a dictionary of model parameters

# client_model = CNN()
# weights = train_local(client_model, client_loaders[0], epochs = 1, lr = 0.1)
# print(weights.keys())

def evaluate_model(dataloader, global_model, is_final):
    # test_loader = DataLoader(mnist_testset, batch_size=64, shuffle=False)
    global_model.eval()
    correct = 0
    total = 0
    with torch.no_grad(): # again, to not compute the graph
        for images, labels in dataloader:
            outputs = global_model(images)
            _, predicted = torch.max(outputs, 1) # the function returns 2 things: the highest val & indices of the highest val
                                                 # by using '_' we tell it to return only the indices and discard the val
            total += labels.size(0) # add batch size to keep track of processed batches
            correct += (predicted == labels).sum().item()
    accuracy = (correct / total)*100
    if(is_final):
        print(f'Accuracy {accuracy}%\n')
    
    
    return accuracy

def aggregate(number_of_samples, client_state_dictionary, global_model):
    global_state_dictionary = {} # here averaged weights will be stored
    total_samples = sum(number_of_samples)
    for key in client_state_dictionary[0].keys(): # here client_state_dictionary[0].keys() is ok because all clients have the same keys:
                                                  # ['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias', 'fully_connected.weight', 'fully_connected.bias']
        global_state_dictionary[key] = torch.zeros_like(client_state_dictionary[0][key]) # initialise with first client
    for client_id, state_dictionary in enumerate(client_state_dictionary): # iterate over clients
        weight = number_of_samples[client_id] / total_samples # we normalise because different clients can have different data size
        for key in state_dictionary.keys(): # iterate over keys of each client
            global_state_dictionary[key] += state_dictionary[key] * weight # eg. global_layer = sum(client_layer * weight)
    # set global model parameters for the next step
    global_model.load_state_dict(global_state_dictionary)
    return global_model


# Data Loaders
# Client abstraction, split training dataset into clients
def create_IID_client_loaders(number_of_clients, mnist_trainset):
    trainset_size = len(mnist_trainset)
    client_trainset_size = trainset_size // number_of_clients  # we use // instead of / to remove the fractional part
    client_trainsets = random_split(mnist_trainset,
                                    [client_trainset_size] * number_of_clients)  # a list with: client_datasets[0], client_datasets[1], ...[5]

    client_training_loaders = []
    for dataset in client_trainsets:
        train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
        client_training_loaders.append(train_loader)
    return client_training_loaders

def create_non_IID_client_loaders(number_of_clients, mnist_trainset):
    # ensure that the 10 digits can be evenly split across clients
    if 10 % number_of_clients != 0:
        raise ValueError(f"number_of_clients={number_of_clients} must divide 10 for non-IID split.")

    # group indices (RELATIVE TO mnist_trainset) by digit label
    label_indices = {i: [] for i in range(10)}

    # If we passed in a Subset (as in FL.py, where mnist_trainset is a train split),
    # we must map from its local indices to labels using the underlying dataset.
    if isinstance(mnist_trainset, Subset):
        base_targets = mnist_trainset.dataset.targets
        for local_idx, original_idx in enumerate(mnist_trainset.indices):
            label = int(base_targets[original_idx])
            label_indices[label].append(local_idx)
    else:
        labels = mnist_trainset.targets  # full MNIST dataset
        for idx, label in enumerate(labels):
            label_indices[int(label)].append(idx)

    # assign digits to clients
    digits_per_client = 10 // number_of_clients
    client_training_loaders = []
    for client in range(number_of_clients):
        client_indices = []
        start_digit = client * digits_per_client
        end_digit = start_digit + digits_per_client
        for digit in range(start_digit, end_digit):
            client_indices.extend(label_indices[digit])
        client_dataset = Subset(mnist_trainset, client_indices)
        loader = DataLoader(client_dataset, batch_size=64, shuffle=True)
        client_training_loaders.append(loader)
    return client_training_loaders


def run_centralised_ml(number_of_epochs, train_loader, val_loader, test_loader, model):
    # do training
    _, losses, val_accuracies = train_local( model, train_loader, val_loader, number_of_epochs, 0.01)
    test_accuracy = evaluate_model(test_loader, model, True)
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    plot_loss( losses)
    plot_acc( val_accuracies)


def run_fl(number_of_rounds, number_of_clients, epochs, global_model, client_training_loaders, val_loader, test_loader):
    global_losses = []  # avg client loss per round
    global_val_accuracies = []  # global model val accuracy per round

    for round in range(number_of_rounds):
        print("---------------------------------")
        print(f"Round {round + 1}:\n")

        client_state_dictionary = []
        number_of_samples = []
        round_losses = []
        # loop over all clients
        for client in range(number_of_clients):
            print(f"Client{client + 1}:")
            local_model = copy.deepcopy(global_model)  # copy global model locally
            weights, losses, _ = train_local(model=local_model, train_loader=client_training_loaders[client],
                                             number_of_epochs=epochs, lr=0.1)  # train it
            client_state_dictionary.append(weights)  # store the weights
            number_of_samples.append(len(client_training_loaders[client].dataset))  # store number of samples
            round_losses.append(losses[-1])
        # aggregate
        round_loss = sum(round_losses) / len(round_losses)
        global_losses.append(round_loss)  # average the loss for the whole round

        global_model = aggregate(number_of_samples, client_state_dictionary, global_model)

        print("Average Loss: " + str(round_loss))
        print("Validation Accuracy:")
        round_val_accuracy = evaluate_model(val_loader, global_model, True)
        global_val_accuracies.append(round_val_accuracy)

    # compute accuracy
    print("Final Model Accuracy: ")
    evaluate_model(test_loader, global_model, True)

    plot_loss(global_losses)
    plot_acc(global_val_accuracies)