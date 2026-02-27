import torch.nn as nn
import torch.optim as optim



def train_local(model, dataloader, epochs, lr):
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.SGD(model.parameters(), lr = lr)
    model.train()
    for epoch in range (epochs):
        running_loss = 0.0
        for images, labels in dataloader:
            # forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # backward pass
            optimiser.zero_grad()
            loss.backward()
            optimiser.step() 
            running_loss += loss.item()
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}\n")
    return model.state_dict() # return a dictionary of model parameters

# client_model = CNN()
# weights = train_local(client_model, client_loaders[0], epochs = 1, lr = 0.1)
# print(weights.keys())