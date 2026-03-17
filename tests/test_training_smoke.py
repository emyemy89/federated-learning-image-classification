import unittest

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, TensorDataset

from src.model_implementation import CNN, MLP
from src.client_training import train_local, run_fl, evaluate_model
from src.data_loading import create_dirichlet_client_loaders


class TrainingSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Create small MNIST subsets for quick smoke tests.
        """
        transform = transforms.ToTensor()
        full_train = datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        # Small fixed subsets for speed
        cls.small_train = Subset(full_train, list(range(256)))
        cls.val = Subset(full_train, list(range(256, 320)))
        cls.test = Subset(full_train, list(range(320, 384)))

    def test_centralised_one_epoch_runs(self):
        """
        Centralized training: ensure one-epoch run completes and returns metrics.
        """
        model = CNN()
        train_loader = DataLoader(self.small_train, batch_size=64, shuffle=True)
        val_loader = DataLoader(self.val, batch_size=64, shuffle=False)
        test_loader = DataLoader(self.test, batch_size=64, shuffle=False)

        state_dict, losses, val_accs = train_local(
            model, train_loader, val_loader, number_of_epochs=1, lr=0.01
        )

        self.assertIsInstance(state_dict, dict)
        self.assertEqual(len(losses), 1)
        self.assertEqual(len(val_accs), 1)
        self.assertGreaterEqual(losses[0], 0.0)
        self.assertGreaterEqual(val_accs[0], 0.0)
        self.assertLessEqual(val_accs[0], 100.0)

    def test_centralised_one_epoch_runs_mlp(self):
        """
        Centralized training with MLP: smoke test that one epoch runs end-to-end.
        """
        model = MLP()
        train_loader = DataLoader(self.small_train, batch_size=64, shuffle=True)
        val_loader = DataLoader(self.val, batch_size=64, shuffle=False)
        test_loader = DataLoader(self.test, batch_size=64, shuffle=False)

        state_dict, losses, val_accs = train_local(
            model, train_loader, val_loader, number_of_epochs=1, lr=0.01
        )

        self.assertIsInstance(state_dict, dict)
        self.assertEqual(len(losses), 1)
        self.assertEqual(len(val_accs), 1)
        self.assertGreaterEqual(losses[0], 0.0)
        self.assertGreaterEqual(val_accs[0], 0.0)
        self.assertLessEqual(val_accs[0], 100.0)

    def test_federated_one_round_runs(self):
        """
        Federated training: ensure a minimal FedAvg run completes and returns metrics.
        """
        number_of_clients = 2
        client_loaders = create_dirichlet_client_loaders(
            number_of_clients=number_of_clients,
            mnist_trainset=self.small_train,
            alpha=0.5,
            batch_size=32,
            seed=0,
        )

        self.assertEqual(len(client_loaders), number_of_clients)

        val_loader = DataLoader(self.val, batch_size=64, shuffle=False)
        test_loader = DataLoader(self.test, batch_size=64, shuffle=False)
        global_model = CNN()

        losses, val_accs = run_fl(
            number_of_rounds=1,
            number_of_clients=number_of_clients,
            epochs=1,
            global_model=global_model,
            client_training_loaders=client_loaders,
            val_loader=val_loader,
            test_loader=test_loader,
            lr = 0.1,
            participation_rate= 0.8,
            convergence="acc",
        )

        self.assertEqual(len(losses), 1)
        self.assertEqual(len(val_accs), 1)
        self.assertGreaterEqual(losses[0], 0.0)
        self.assertGreaterEqual(val_accs[0], 0.0)
        self.assertLessEqual(val_accs[0], 100.0)

    def test_f1_convergence_triggers_stop_flag(self):
        """
        Evaluate_model: ensures F1-based convergence flag behaves as expected.
        """

        class ConstantModel(torch.nn.Module):
            def __init__(self, num_classes: int = 10):
                super().__init__()
                self.logits = torch.nn.Parameter(torch.zeros(num_classes), requires_grad=False)

            def forward(self, x):
                # broadcast fixed logits to batch size
                batch_size = x.shape[0]
                return self.logits.unsqueeze(0).expand(batch_size, -1)

        # Small synthetic dataset: all zeros images, fixed labels
        inputs = torch.zeros(32, 1, 28, 28)
        labels = torch.zeros(32, dtype=torch.long)
        dataset = TensorDataset(inputs, labels)
        loader = DataLoader(dataset, batch_size=16, shuffle=False)

        model = ConstantModel(num_classes=10)

        # First call establishes a baseline F1
        acc1, stop1, f1_1 = evaluate_model(loader, model, prev_f1=0.0, prev_accuracy=0, convergence="acc")
        self.assertFalse(stop1)

        # Second call with nearly identical F1 should trigger convergence
        acc2, stop2, f1_2 = evaluate_model(loader, model, prev_f1=f1_1, prev_accuracy=acc1, convergence="acc")
        self.assertEqual(acc1, acc2)
        self.assertAlmostEqual(f1_1, f1_2, places=5)
        self.assertTrue(stop2)


if __name__ == "__main__":
    unittest.main()

