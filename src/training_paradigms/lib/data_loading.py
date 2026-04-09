from collections import Counter

import torch
from torch.utils.data import random_split, DataLoader, Subset


# Data Loaders
# Client abstraction, split training dataset into clients

# loader for IID data
def create_IID_client_loaders(number_of_clients, mnist_trainset):
    trainset_size = len(mnist_trainset)
    client_trainset_size = trainset_size // number_of_clients  # we use // instead of / to remove the fractional part
    client_trainsets = random_split(mnist_trainset,
                                    [client_trainset_size] * number_of_clients)  # a list with: client_datasets[0], client_datasets[1], ...[5]
    client_training_loaders = []
    for dataset in client_trainsets:
        train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
        client_training_loaders.append(train_loader)
    return client_training_loaders

# loader for pure non-IID data
def create_non_IID_client_loaders(number_of_clients, mnist_trainset):
    if number_of_clients <= 0:
        raise ValueError("number_of_clients must be > 0")
    # group indices (RELATIVE TO mnist_trainset) by digit label
    label_indices = {i: [] for i in range(10)}
    # If we passed in a Subset (as in FL.py, where mnist_trainset is a train split),
    # we must map from its local indices to labels using the underlying dataset.
    if isinstance(mnist_trainset, Subset):
        base_targets = mnist_trainset.dataset.targets
        for local_idx, original_idx in enumerate(mnist_trainset.indices):
            label = int(base_targets[original_idx])
            label_indices[label].append(local_idx)
    else:
        labels = mnist_trainset.targets  # full MNIST dataset
        for idx, label in enumerate(labels):
            label_indices[int(label)].append(idx)
    # assign digits to clients (works for any number_of_clients)
    digits = list(range(10))
    q, r = divmod(len(digits), number_of_clients)  # base digits per client + remainder
    client_training_loaders = []
    for client in range(number_of_clients):
        if q == 0:
            # more clients than digits: reuse digits round-robin so every client has data
            assigned_digits = [digits[client % len(digits)]]
        else:
            start = client * q + min(client, r)
            end = start + q + (1 if client < r else 0)
            assigned_digits = digits[start:end]
        client_indices = []
        for digit in assigned_digits:
            client_indices.extend(label_indices[digit])
        client_dataset = Subset(mnist_trainset, client_indices)
        loader = DataLoader(client_dataset, batch_size=64, shuffle=True)
        client_training_loaders.append(loader)
    return client_training_loaders

# loader for dirichlet distribution of non-IID data
def create_dirichlet_client_loaders(number_of_clients, mnist_trainset, alpha=0.5, batch_size=64, seed=None):
    """
    Non-IID split using Dirichlet distribution over class proportions.
    - Smaller alpha => more skewed per-client label distributions.
    - Larger alpha  => more balanced per-client label distributions.
    """
    if number_of_clients <= 0:
        raise ValueError("number_of_clients must be > 0")
    if alpha <= 0:
        raise ValueError("alpha must be > 0")
    if seed is not None:
        torch.manual_seed(seed)
    # Build label -> indices mapping (indices relative to mnist_trainset)
    label_indices = {i: [] for i in range(10)} # dictionary of digits
    if isinstance(mnist_trainset, Subset):
        base_targets = mnist_trainset.dataset.targets
        for local_idx, original_idx in enumerate(mnist_trainset.indices):
            label = int(base_targets[original_idx])
            label_indices[label].append(local_idx)
    else:
        labels = mnist_trainset.targets
        for idx, label in enumerate(labels):
            label_indices[int(label)].append(idx)
    # Allocate samples per class to clients according to Dirichlet proportions
    client_indices = [[] for _ in range(number_of_clients)] # list of size 'number_of_clients'
    dirichlet = torch.distributions.Dirichlet(torch.full((number_of_clients,), float(alpha))) # create the distribution based on alpha
    for digit in range(10): # loops over the 10 digits
        idxs = label_indices[digit]
        if not idxs:
            continue
        # shuffle indices for this class
        perm = torch.randperm(len(idxs)).tolist() # shuffle the indices so distribution is random
        idxs = [idxs[i] for i in perm]
        proportions = dirichlet.sample() # sample proportions, e.g. for 5 clients: [0.4, 0.3, 0.2, 0.07, 0.03]
        cut_points = (torch.cumsum(proportions, dim=0) * len(idxs)).to(torch.int64) # convert the proportions above to counts(how many samples)
        cut_points[-1] = len(idxs)  # avoid any rounding issues

        start = 0
        for client_id, end in enumerate(cut_points.tolist()):
            if end > start:
                client_indices[client_id].extend(idxs[start:end]) # assign those counts to clients
            start = end

    # Avoid empty clients (can happen with small alpha / unlucky sampling)
    empties = [i for i, idxs in enumerate(client_indices) if len(idxs) == 0]
    for empty_client in empties:
        donor = max(range(number_of_clients), key=lambda i: len(client_indices[i])) # if a client has no samples, another clients with many classes can donate one to the client without
        if len(client_indices[donor]) == 0:
            break
        client_indices[empty_client].append(client_indices[donor].pop())
    return [
        DataLoader(Subset(mnist_trainset, idxs), batch_size=batch_size, shuffle=True)
        for idxs in client_indices
    ]

# helper function for printing what digits do clients get when data is non-IID
def debug_non_iid_split(client_training_loaders):
    print("\nSampled label distribution per client (first few batches):")
    for client_id, loader in enumerate(client_training_loaders):
        label_counts = Counter()
        for batch_idx, (_, labels) in enumerate(loader):
            label_counts.update(labels.tolist())
            if batch_idx >= 9:  # sample first 10 batches only
                break
        digits = sorted(label_counts.keys())
        print(
            f"Client {client_id}: digits={digits} "
            f"counts={{" + ", ".join(f"{d}:{label_counts[d]}" for d in digits) + "}}"
        )