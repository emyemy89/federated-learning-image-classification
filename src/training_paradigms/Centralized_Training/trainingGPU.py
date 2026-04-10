import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def train_model(train_loader, model, device):
    # loading,
                                                                              # shuffle is true so the model has a general view, 
                                                                              # adds randomness basically
    # Loss function and optimiser
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9) # here the learning rate is 0.001, 
                                                                          # momentum is sth like "if we go in the correct dir, let's go faster"
    
    # The training loop for a few epochs
    number_of_epochs = 3
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
            loss.backward() # computes gradient
            optimiser.step() # takes a step in the direction of gradient
    
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{number_of_epochs}, Average Loss: {avg_loss:.4f}")