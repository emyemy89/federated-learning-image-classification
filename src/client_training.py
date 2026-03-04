import torch
import torch.nn as nn
import torch.optim as optim



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
            val_accuracy = evaluate_model(val_loader, model)
            val_accuracies.append(val_accuracy)
            print(f"Epoch {epoch+1}/{number_of_epochs}, Loss: {avg_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        else:
            print(f"Epoch {epoch+1}/{number_of_epochs}, Loss: {avg_loss:.4f}")
    return model.state_dict(), losses, val_accuracies # return a dictionary of model parameters

# client_model = CNN()
# weights = train_local(client_model, client_loaders[0], epochs = 1, lr = 0.1)
# print(weights.keys())

def evaluate_model(dataloader, global_model):
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