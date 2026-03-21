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
    plt.xticks(get_sparse_ticks(len(losses)))
    plt.tight_layout()
    plt.savefig('loss.png')
    plt.show()
    


def plot_acc( val_accuracies):
    epochs = range(1, len(val_accuracies) + 1)
    #acc plot
    plt.figure()
    plt.plot(epochs, val_accuracies, marker='o', color='orange')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.xticks(get_sparse_ticks(len(val_accuracies)))
    plt.tight_layout()
    plt.savefig('accuracy.png')
    plt.show()




def plot_loss_overlay(central_losses, fed_results):
    """
    Overlay loss curves for centralized vs federated training.
    central_losses: list of per-epoch losses from centralized training
    fed_losses:     list of per-round losses from federated training
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(central_losses, label="Centralized", marker=".", linewidth=2)
    #ax.plot(fed_losses,     label="Federated (FedAvg)", marker="s", linewidth=2, linestyle="--")
    # Federated curves (multiple)
    colors = ["orange", "green", "red", "purple", "brown"]
    for i, (num_clients, results) in enumerate(fed_results.items()):
        color = colors[i % len(colors)]
        ax.plot(
            results["losses"],
            label=f"FL ({num_clients} clients)",
            linestyle="solid",
            marker=".",
            color=color,
            linewidth=1.5
        )

    ax.set_xlabel("Round Number")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss: Centralized vs Federated")
    ax.legend()
    ax.grid(True, alpha=0.3)
    max_rounds = max(len(central_losses), *(len(r["losses"]) for r in fed_results.values()))
    ax.set_xticks(get_sparse_ticks(max_rounds))
    plt.tight_layout()
    plt.savefig("loss_comparison.png", dpi=150)
    plt.show()


def plot_acc_overlay(central_accs, fed_results):
    """
    Overlay validation accuracy curves for centralized vs federated training.
    central_accs: list of per-epoch val accuracies from centralized training
    fed_accs:     list of per-round val accuracies from federated training
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(central_accs, label="Centralized", marker=".", linewidth=2)
    #ax.plot(fed_accs,     label="Federated (FedAvg)", marker="s", linewidth=2, linestyle="--")
    colors = ["orange", "green", "red", "purple", "brown"]
    for i,(num_clients, results) in enumerate(fed_results.items()):
        color = colors[i % len(colors)]
        ax.plot(
            results["accs"],
            label=f"FL ({num_clients} clients)",
            linestyle="solid",
            marker=".",
            color=color,
            linewidth=1.5
        )

    ax.set_xlabel("Round Number")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy per round: Centralized vs Federated")
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    max_rounds = max(len(central_accs), *(len(r["losses"]) for r in fed_results.values()))
    ax.set_xticks(get_sparse_ticks(max_rounds))
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

def get_sparse_ticks(n, num_ticks=5):
    """
    Returns a list of x-axis ticks to display.
    Always includes 1 and the last round.
    n: total number of rounds
    num_ticks: approx number of ticks to show
    """
    if n <= num_ticks:
        return list(range(1, n+1))
    else:
        # Evenly spaced ticks
        ticks = [1]
        # generate num_ticks-2 intermediate ticks
        for t in range(1, num_ticks-1):
            ticks.append(round(t * (n-1)/(num_ticks-1)) + 1)
        ticks.append(n)
        # remove duplicates and sort
        ticks = sorted(list(set(ticks)))
        return ticks