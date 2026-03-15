from collections import Counter
import copy
import torch
import time
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, f1_score

from src.plotting import plot_loss, plot_acc, plot_cm

# function for local training in FL settings
def train_local(model, train_loader, val_loader = None, number_of_epochs = 1, lr = 0.1):
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.SGD(model.parameters(), lr = lr)
    model.train()
    losses, val_accuracies = [], []
    prev_f1 = 0
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
            val_accuracy, stop, current_f1 = evaluate_model(val_loader, model, prev_f1)
            prev_f1 = current_f1
            if stop:
                print(f"Convergence reached at epoch {epoch + 1}. Stopping training.")
                break
            val_accuracies.append(val_accuracy)
            print(f"Epoch {epoch+1}/{number_of_epochs}, Loss: {avg_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        else:
            print(f"Epoch {epoch+1}/{number_of_epochs}, Loss: {avg_loss:.4f}")
    return model.state_dict(), losses, val_accuracies # return a dictionary of model parameters


# function for calculating accuracy
def evaluate_model(dataloader, global_model, prev_f1):
    global_model.eval()
    correct = 0
    total = 0
    stop = False
    all_true = []
    all_predicted = []
    with torch.no_grad(): # again, to not compute the graph
        for images, labels in dataloader:
            outputs = global_model(images)
            _, predicted = torch.max(outputs, 1) # the function returns 2 things: the highest val & indices of the highest val
                                                 # by using '_' we tell it to return only the indices and discard the val
            total += labels.size(0) # add batch size to keep track of processed batches
            correct += (predicted == labels).sum().item()
            all_true.extend(labels.cpu().numpy()) # we need to convert labels to numpy arrays (from tensors) but they need to be on the CPU first (GPU doesn't allow)
            all_predicted.extend(predicted.cpu().numpy())
    accuracy = (correct / total)*100
    current_f1 = f1_score(all_true, all_predicted, average='macro')
    print(f"F1 difference thingy is {current_f1- prev_f1}\n")
    if (current_f1-prev_f1 < 0.001):
        stop = True
        generate_confusion_matrix(all_true, all_predicted)
        print(f'Accuracy {accuracy}%\n')
    return accuracy, stop, current_f1

# function to aggregate weights in FL global model computation at each round
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


# helper function for running ML algorithms
def run_centralised_ml(number_of_epochs, train_loader, val_loader, test_loader, model):
    start_time = time.time()
    # do training
    _, losses, val_accuracies = train_local( model, train_loader, val_loader, number_of_epochs, 0.01)
    #test_accuracy, _, _ = evaluate_model(test_loader, model, True, False)
    end_time = time.time()
    print(f"Total training time: {end_time-start_time:.2f} seconds")
    #print(f"Test Accuracy: {test_accuracy:.2f}%")
    plot_loss( losses)
    plot_acc( val_accuracies)
    return losses, val_accuracies

# helper function for running FL algorithms
def run_fl(number_of_rounds, number_of_clients, epochs, global_model, client_training_loaders, val_loader, test_loader):
    global_losses = []  # avg client loss per round
    global_val_accuracies = []  # global model val accuracy per round
    start_time = time.time()
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
            round_losses.append(sum(losses) / len(losses))  # avg over epochs for this client
        # aggregate
        round_loss = sum(round_losses) / len(round_losses)
        global_losses.append(round_loss)  # average the loss for the whole round

        global_model = aggregate(number_of_samples, client_state_dictionary, global_model)

        print("Average Loss: " + str(round_loss))
        print("Validation Accuracy:")
        round_val_accuracy, _, _ = evaluate_model(val_loader, global_model, True, False)
        global_val_accuracies.append(round_val_accuracy)
    # compute accuracy
    print("Final Model Accuracy: ")
    evaluate_model(test_loader, global_model, True, True, 0)
    end_time = time.time()
    print(f"Total training time: {end_time-start_time:.2f} seconds")

    plot_loss(global_losses)
    plot_acc(global_val_accuracies)
    return global_losses, global_val_accuracies


# Generates a confusion matrix
def generate_confusion_matrix(true, predicted):
    cm = confusion_matrix(true, predicted)
    plot_cm(cm)
    return cm

