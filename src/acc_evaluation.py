import torch


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