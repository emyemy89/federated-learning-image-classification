import matplotlib.pyplot as plt

def plot_loss( losses):
    epochs = range(1, len(losses) + 1)

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
    


def plot_acc( val_accuracies):
    epochs = range(1, len(val_accuracies) + 1)
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




def plot_loss_overlay(central_losses, fed_losses):
    """
    Overlay loss curves for centralized vs federated training.
    central_losses: list of per-epoch losses from centralized training
    fed_losses:     list of per-round losses from federated training
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(central_losses, label="Centralized", marker="o", linewidth=2)
    ax.plot(fed_losses,     label="Federated (FedAvg)", marker="s", linewidth=2, linestyle="--")

    ax.set_xlabel("Epoch / Round")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss: Centralized vs Federated")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("loss_comparison.png", dpi=150)
    plt.show()


def plot_acc_overlay(central_accs, fed_accs):
    """
    Overlay validation accuracy curves for centralized vs federated training.
    central_accs: list of per-epoch val accuracies from centralized training
    fed_accs:     list of per-round val accuracies from federated training
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(central_accs, label="Centralized", marker="o", linewidth=2)
    ax.plot(fed_accs,     label="Federated (FedAvg)", marker="s", linewidth=2, linestyle="--")

    ax.set_xlabel("Epoch / Round")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Validation Accuracy: Centralized vs Federated")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("acc_comparison.png", dpi=150)
    plt.show()