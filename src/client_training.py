import random
from collections import Counter
import copy
import torch
import time
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, f1_score

from src.plotting import plot_loss, plot_acc, plot_cm
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

def train_local(model, train_loader, val_loader = None, number_of_epochs = 1, lr = 0.1, convergence = None):
    """
    Locally trains ``model`` on ``train_loader`` (cross-entropy, SGD); validates when ``val_loader`` is provided.
    Parameters — model: ``nn.Module``. train_loader: training ``DataLoader``. val_loader: optional ``DataLoader``.
    number_of_epochs: ``int``. lr: ``float`` SGD step size. convergence: ``"f1"``, ``"acc"``, or ``None`` for early-stop.
    Returns: ``tuple`` — (``state_dict``, per-epoch training losses ``list[float]``, per-epoch val accuracies ``list[float]``).
    """
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.SGD(model.parameters(), lr = lr)
    model.train()
    losses, val_accuracies = [], []
    prev_f1 = 0
    prev_accuracy = 0
    num_improve_rounds = 0
    patience = 4
    model.to(device)
    for epoch in range (number_of_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
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
            val_accuracy, small_change, current_f1, all_true, all_predicted = evaluate_model(val_loader, model, prev_f1, prev_accuracy, convergence = convergence)
            prev_f1 = current_f1
            prev_accuracy = val_accuracy
            if small_change:
                num_improve_rounds += 1
            else:
                num_improve_rounds = 0
            if num_improve_rounds >= patience:
                print(f"Convergence reached at FL round {epoch + 1}. Stopping training.")
                generate_confusion_matrix(all_true, all_predicted)
                break
            val_accuracies.append(val_accuracy)
            print(f"Epoch {epoch+1}/{number_of_epochs}, Loss: {avg_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        else:
            print(f"Epoch {epoch+1}/{number_of_epochs}, Loss: {avg_loss:.4f}")
    return model.state_dict(), losses, val_accuracies # return a dictionary of model parameters


def evaluate_model(dataloader, global_model, prev_f1, prev_accuracy, convergence):
    """
    Evaluates ``global_model`` on ``dataloader``; computes accuracy, macro-F1, and optional convergence signal.
    Parameters — dataloader: ``DataLoader``. global_model: ``nn.Module``. prev_f1, prev_accuracy: prior metrics ``float``.
    convergence: ``"f1"``, ``"acc"``, or ``None`` (gates ``small_change`` thresholds).
    Returns: ``tuple`` — (accuracy ``float`` %, ``small_change`` ``bool``, ``current_f1`` ``float``, ``all_true`` ``list``, ``all_predicted`` ``list``).
    """
    global_model.eval()
    correct = 0
    total = 0
    #stop = False
    small_change = False
    all_true = []
    all_predicted = []
    global_model.to(device)
    with torch.no_grad(): # again, to not compute the graph
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = global_model(images)
            _, predicted = torch.max(outputs, 1) # the function returns 2 things: the highest val & indices of the highest val
                                                 # by using '_' we tell it to return only the indices and discard the val
            total += labels.size(0) # add batch size to keep track of processed batches
            correct += (predicted == labels).sum().item()
            all_true.extend(labels.cpu().numpy()) # we need to convert labels to numpy arrays (from tensors) but they need to be on the CPU first (GPU doesn't allow)
            all_predicted.extend(predicted.cpu().numpy())
    accuracy = (correct / total)*100
    current_f1 = f1_score(all_true, all_predicted, average='macro')
    # print(f"F1 difference thingy is {current_f1- prev_f1}\n")
    if (convergence == "f1" and abs(current_f1 - prev_f1) < 0.0001) or \
   (convergence == "acc" and abs(accuracy - prev_accuracy) < 0.05):
        small_change = True
        #generate_confusion_matrix(all_true, all_predicted)
        print(f'Accuracy {accuracy}%\n')
    return accuracy, small_change, current_f1, all_true, all_predicted


def aggregate(number_of_samples, client_state_dictionary, global_model):
    """
    Federated averaging: sample-weighted mean of client ``state_dict`` entries written into ``global_model``.
    Parameters — number_of_samples: ``list[int]`` per-client sample counts. client_state_dictionary: ``list`` of ``dict`` (tensor weights).
    global_model: ``nn.Module`` updated in-place via ``load_state_dict``.
    Returns: ``global_model`` ``nn.Module``.
    """
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


def run_centralised_ml(number_of_epochs, train_loader, val_loader, test_loader, model, lr, convergence):
    """
    Centralized training via ``train_local``; prints timing and test accuracy; plots loss and validation accuracy curves.
    Parameters — number_of_epochs: ``int``. train_loader, val_loader, test_loader: ``DataLoader``. model: ``nn.Module``. lr: ``float``.
    convergence: ``"f1"``, ``"acc"``, or ``None``.
    Returns: ``tuple`` — (training losses ``list[float]``, validation accuracies ``list[float]``).
    """
    start_time = time.time()
    model = model.to(device)
    # do training
    _, losses, val_accuracies = train_local( model, train_loader, val_loader, number_of_epochs, lr, convergence)
    end_time = time.time()
    # if no convergence
    test_accuracy, _, _, _, _ = evaluate_model(test_loader, model, 0, 0, None)
    print(f"Total training time: {end_time-start_time:.2f} seconds")
    print(f"No convergence, but test Accuracy: {test_accuracy:.2f}%")
    plot_loss( losses)
    plot_acc( val_accuracies)
    return losses, val_accuracies


def run_fl(number_of_rounds, number_of_clients, epochs, global_model, client_training_loaders, val_loader, test_loader, lr, participation_rate, convergence):
    """
    Federated rounds: random client subset, local ``train_local``, ``aggregate`` (FedAvg), validate, optional early stop; plots metrics.
    Parameters — number_of_rounds, number_of_clients, epochs: ``int``. global_model: ``nn.Module``. client_training_loaders: ``list[DataLoader]``.
    val_loader, test_loader: ``DataLoader``. lr: ``float``. participation_rate: ``float``. convergence: ``"f1"``, ``"acc"``, or ``None``.
    Returns: ``tuple`` — (per-round mean client losses ``list[float]``, per-round validation accuracies ``list[float]``).
    """
    global_losses = []  # avg client loss per round
    global_val_accuracies = []  # global model val accuracy per round
    start_time = time.time()
    prev_f1 = 0
    prev_accuracy = 0
    num_improve_rounds = 0
    patience = 4
    for round in range(number_of_rounds):
        print("---------------------------------")
        print(f"Round {round + 1}:\n")
        client_state_dictionary = []
        number_of_samples = []
        round_losses = []
        # partial client participation
        sample_of_clients = max(1, int(number_of_clients * participation_rate))
        selected_clients = random.sample(range(number_of_clients), sample_of_clients)
        # loop over all clients
        for client in selected_clients:
            print(f"Client{client + 1}:")
            local_model = copy.deepcopy(global_model).to(device)  # copy global model locally
            weights, losses, _ = train_local(model=local_model, train_loader=client_training_loaders[client],
                                             number_of_epochs=epochs, lr=lr, convergence = convergence)  # train it
            client_state_dictionary.append(weights)  # store the weights
            number_of_samples.append(len(client_training_loaders[client].dataset))  # store number of samples
            round_losses.append(sum(losses) / len(losses))  # avg over epochs for this client
        # aggregate
        round_loss = sum(round_losses) / len(round_losses)
        global_losses.append(round_loss)  # average the loss for the whole round

        global_model = aggregate(number_of_samples, client_state_dictionary, global_model)

        print("Average Loss: " + str(round_loss))
        round_val_accuracy, small_change, current_f1, all_true, all_predicted = evaluate_model(val_loader, global_model, prev_f1, prev_accuracy, convergence)
        print(f"Validation Accuracy: {round_val_accuracy:.2f}")
        if small_change:
            num_improve_rounds += 1
        else:
            num_improve_rounds = 0
        if num_improve_rounds >= patience:
            print(f"Convergence reached at FL round {round + 1}. Stopping training.")
            generate_confusion_matrix(all_true, all_predicted)  # optional move here
            break
        prev_f1 = current_f1
        prev_accuracy = round_val_accuracy
        global_val_accuracies.append(round_val_accuracy)
    # compute accuracy if no convergence
    print("Not converged, but Final Model Accuracy: ")
    evaluate_model(test_loader, global_model, 0,0, None)
    end_time = time.time()
    print(f"Total training time: {end_time-start_time:.2f} seconds")

    plot_loss(global_losses)
    plot_acc(global_val_accuracies)
    return global_losses, global_val_accuracies


def generate_confusion_matrix(true, predicted):
    """
    Computes a confusion matrix from label sequences and displays it via ``plot_cm``.
    Parameters — true: ground-truth class indices (sequence). predicted: predicted class indices (sequence, aligned with ``true``).
    Returns: ``numpy.ndarray`` from ``sklearn.metrics.confusion_matrix``.
    """
    cm = confusion_matrix(true, predicted)
    plot_cm(cm)
    return cm

