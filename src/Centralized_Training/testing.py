import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def model_testing(test_loader, model):
    model.eval() # put it in evaluation mode, good practice
    correct = 0
    total = 0
    with torch.no_grad(): # don't compute gradients so we don't store and consume memory(no need in testing), faster evaluation
        for images, labels in test_loader:
            images = images
            labels = labels
            outputs = model(images)
            _, predicted = torch.max(outputs, 1) # the function returns 2 things: the highest val & indices of the highest val
                                                 # by using '_' we tell it to return only the indices and discard the val
            total += labels.size(0) # add batch size to keep track of processed batches
            correct += (predicted == labels).sum().item()
    accuracy = correct/total
    return accuracy
    #print(f'Test accuracy: {accuracy * 100:.2f}%')