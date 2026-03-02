import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import testing 
from testing import model_testing

def train_model(number_of_epochs, train_loader, val_loader, model, lr, momentum):
    # loading,
                                                                              # shuffle is true so the model has a general view, 
                                                                              # adds randomness basically
    # Loss function and optimiser
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.SGD(model.parameters(), lr, momentum) # here the learning rate is 0.001, 
                                                                          # momentum is sth like "if we go in the correct dir, let's go faster"
    
    # The training loop for a few epochs
    for epoch in range (number_of_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images = images
            labels = labels
            # forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # backward pass
            optimiser.zero_grad()
            loss.backward() # computes gradient
            optimiser.step() # takes a step in the direction of gradient
    
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        val_accuracy = model_testing(val_loader, model)
        print(f'Validation accuracy: {val_accuracy * 100:.2f}%')
        print(f"Epoch {epoch+1}/{number_of_epochs}, Average Loss: {avg_loss:.4f}")