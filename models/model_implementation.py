# MLP source: https://machinelearningmastery.com/building-multilayer-perceptron-models-in-pytorch/
import torch
import torch.nn as nn

# CNN
# a CNN with 2 layers: Layer1-> Relu->Pool->Layer2->...->Flatten->Fully connected->Output
class CNN(nn.Module): # we take the standard class and modify it
    def __init__(self): # this is a constructor
        super(CNN,self).__init__()
        # First layer
        self.conv1 = nn.Conv2d(in_channels=1,   # Mnist has 1 channel
                          out_channels=16, # 16 filters
                          kernel_size=3)   # 3x3 kernel size
        # Second layer
        self.conv2 = nn.Conv2d(in_channels=16,   # Mnist has 1 channel
                          out_channels=32, # 16 filters
                          kernel_size=3)   # 3x3 kernel size
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        # create the fully connected layer
        self.fully_connected = nn.Linear(800,10)

    def forward(self,x):
        x = self.pool(self.relu(self.conv1(x))) # do relu and pooling for layer 1
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1) # here we reshape the tensor to be of 'out.size(0)' rows 
                                  # but don't know exactly how many columns, so we specify '-1'
                                  # [1, 32, 5, 5] -> [1, 800]
        x = self.fully_connected(x)
        return x

# image is treated as a vector of numbers
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128), # a linear layer computes y=Wx+b (x is input vector, W is weight matrix and b is bias)
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        return self.network(x)