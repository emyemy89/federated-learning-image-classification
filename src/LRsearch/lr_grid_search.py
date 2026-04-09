import sys
import os
from typing import Dict, List, Tuple, Any

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Ensure project root is on the path when running this file directly
sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("../.."))

from src.utils_client_training import run_fl, run_centralised_ml
from src.data_loading import create_dirichlet_client_loaders
from models.model_implementation import CNN


def prepare_data(
    batch_size: int,
    number_of_clients: int,
    dirichlet_alpha: float,
) -> Tuple[
    List[DataLoader],
    DataLoader,
    DataLoader,
    DataLoader,
]:
    """
    Prepare MNIST data loaders shared across all runs.
    """
    transform = transforms.ToTensor()

    full_mnist_trainset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )
    train_size = int(0.7 * len(full_mnist_trainset))
    val_size = len(full_mnist_trainset) - train_size

    mnist_trainset, mnist_valset = random_split(
        full_mnist_trainset, [train_size, val_size]
    )
    mnist_testset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    client_training_loaders = create_dirichlet_client_loaders(
        number_of_clients=number_of_clients,
        mnist_trainset=mnist_trainset,
        alpha=dirichlet_alpha,
    )

    train_loader = DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(mnist_valset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(mnist_testset, batch_size=batch_size, shuffle=False)

    return client_training_loaders, train_loader, val_loader, test_loader


def run_centralised_lr_grid_search(
    lrs: List[float],
    number_of_epochs: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
) -> Dict[float, Dict[str, Any]]:
    """
    Run centralised training for each learning rate in lrs.
    Returns a dict mapping lr -> result metrics.
    """
    results: Dict[float, Dict[str, Any]] = {}

    for lr in lrs:
        print("=" * 60)
        print(f"Centralised training with lr = {lr}")
        print("=" * 60)

        model = CNN()
        losses, val_accs = run_centralised_ml(
            number_of_epochs=number_of_epochs,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            model=model,
            lr=lr,
        )

        final_val_acc = val_accs[-1] if len(val_accs) > 0 else None
        results[lr] = {
            "losses": losses,
            "val_accs": val_accs,
            "final_val_acc": final_val_acc,
        }

    return results


def run_fl_lr_grid_search(
    lrs: List[float],
    number_of_rounds: int,
    number_of_clients: int,
    epochs_per_client: int,
    client_training_loaders: List[DataLoader],
    val_loader: DataLoader,
    test_loader: DataLoader,
    participation_rate: float = 1.0,
) -> Dict[float, Dict[str, Any]]:
    """
    Run federated learning for each learning rate in lrs.
    Returns a dict mapping lr -> result metrics.
    """
    results: Dict[float, Dict[str, Any]] = {}

    for lr in lrs:
        print("=" * 60)
        print(f"Federated learning with lr = {lr}")
        print("=" * 60)

        global_model = CNN()
        losses, val_accs = run_fl(
            number_of_rounds=number_of_rounds,
            number_of_clients=number_of_clients,
            epochs=epochs_per_client,
            global_model=global_model,
            client_training_loaders=client_training_loaders,
            val_loader=val_loader,
            test_loader=test_loader,
            lr=lr,
            participation_rate=participation_rate,
        )

        final_val_acc = val_accs[-1] if len(val_accs) > 0 else None
        results[lr] = {
            "losses": losses,
            "val_accs": val_accs,
            "final_val_acc": final_val_acc,
        }

    return results


def main() -> None:
    # Shared hyperparameters
    NUMBER_OF_CLIENTS = 5
    NUMBER_OF_EPOCHS_PER_CLIENT = 3
    NUMBER_OF_ROUNDS = 10
    DIRICHLET_ALPHA = 1.0
    BATCH_SIZE = 64

    # Learning rates to explore
    centralised_lrs = [0.01, 0.03]
    fl_lrs = [0.01, 0.05, 0.1, 0.5]

    # Prepare data once and reuse for all runs
    (
        client_training_loaders,
        train_loader,
        val_loader,
        test_loader,
    ) = prepare_data(
        batch_size=BATCH_SIZE,
        number_of_clients=NUMBER_OF_CLIENTS,
        dirichlet_alpha=DIRICHLET_ALPHA,
    )

    # Centralised LR grid search
    centralised_results = run_centralised_lr_grid_search(
        lrs=centralised_lrs,
        number_of_epochs=NUMBER_OF_ROUNDS,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )

    # Federated LR grid search
    fl_results = run_fl_lr_grid_search(
        lrs=fl_lrs,
        number_of_rounds=NUMBER_OF_ROUNDS,
        number_of_clients=NUMBER_OF_CLIENTS,
        epochs_per_client=NUMBER_OF_EPOCHS_PER_CLIENT,
        client_training_loaders=client_training_loaders,
        val_loader=val_loader,
        test_loader=test_loader,
        participation_rate=1.0,
    )

    # Simple text summary of results
    print("\n" + "#" * 60)
    print("Centralised LR grid search summary")
    print("#" * 60)
    for lr, metrics in centralised_results.items():
        print(
            f"LR={lr}: final val acc = {metrics['final_val_acc']:.4f}"
            if metrics["final_val_acc"] is not None
            else f"LR={lr}: final val acc = None"
        )

    print("\n" + "#" * 60)
    print("Federated LR grid search summary")
    print("#" * 60)
    for lr, metrics in fl_results.items():
        print(
            f"LR={lr}: final val acc = {metrics['final_val_acc']:.4f}"
            if metrics["final_val_acc"] is not None
            else f"LR={lr}: final val acc = None"
        )


if __name__ == "__main__":
    # Do not run anything automatically when imported
    main()

