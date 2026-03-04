import matplotlib.pyplot as plt

def plot_loss(number_of_epochs, losses):
    epochs = range(1, number_of_epochs + 1)

#loss plot
    plt.figure()
    plt.plot(epochs, losses, marker='o')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(epochs)
    plt.tight_layout()
    plt.savefig('loss.png')
    plt.show()
    


def plot_acc(number_of_epochs, val_accuracies):
    epochs = range(1, number_of_epochs + 1)
    #acc plot
    plt.figure()
    plt.plot(epochs, val_accuracies, marker='o', color='orange')
    plt.title('Validation Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.xticks(epochs)
    plt.tight_layout()
    plt.savefig('accuracy.png')
    plt.show()

