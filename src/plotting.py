import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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


def plot_client_label_distribution(client_training_loaders):
    """
    Visualize how many samples of each label each client has
    using a heatmap (clients x classes).
    """
    num_clients = len(client_training_loaders)
    num_classes = 10
    counts = np.zeros((num_clients, num_classes), dtype=int)

    for client_id, loader in enumerate(client_training_loaders):
        dataset = loader.dataset
        for _, label in dataset:
            counts[client_id, int(label)] += 1

    fig, ax = plt.subplots(figsize=(1 + num_classes * 0.6, 1 + num_clients * 0.6))
    im = ax.imshow(counts, aspect="auto", cmap="Blues")

    ax.set_xlabel("Digit label")
    ax.set_ylabel("Client")
    ax.set_title("Client label distribution (Dirichlet)")

    ax.set_xticks(range(num_classes))
    ax.set_xticklabels(range(num_classes))
    ax.set_yticks(range(num_clients))
    ax.set_yticklabels([f"C{idx}" for idx in range(num_clients)])

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Number of samples")

    plt.tight_layout()
    plt.savefig("client_label_distribution.png", dpi=150)
    plt.show()

def plot_cm(cm):
    plt.figure()
    sns.heatmap(
        cm,
        annot = True,
        fmt = "d",  # force integer formatting
        cmap = "Blues",
        annot_kws = {"size": 10}  # control number size
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()