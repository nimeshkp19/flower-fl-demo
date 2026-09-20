"""fl_demo: model, non-IID data partitioning, and local train/test loops."""

from collections import Counter
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import PathologicalPartitioner


# -----------------------------------------------------------------------
# Model: a small CNN for MNIST (1 input channel, 10 output classes)
# -----------------------------------------------------------------------
class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 -> 14x14
        x = self.pool(F.relu(self.conv2(x)))  # 14x14 -> 7x7
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# -----------------------------------------------------------------------
# Data: MNIST split into deliberately non-IID partitions.
# Each simulated client only ever sees `classes_per_partition` digits.
# -----------------------------------------------------------------------
CLASSES_PER_PARTITION = 2

fds: Optional[FederatedDataset] = None  # cached across calls in the same process


def _get_federated_dataset(num_partitions: int) -> FederatedDataset:
    global fds
    if fds is None:
        partitioner = PathologicalPartitioner(
        num_partitions=num_partitions,
        partition_by="label",
        num_classes_per_partition=CLASSES_PER_PARTITION,
        class_assignment_mode="deterministic",
        )
        fds = FederatedDataset(
            dataset="ylecun/mnist",
            partitioners={"train": partitioner},
        )
    return fds


def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """Load one non-IID partition of MNIST, split into train/test DataLoaders."""
    dataset = _get_federated_dataset(num_partitions)
    partition = dataset.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    pytorch_transforms = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    def apply_transforms(batch):
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=batch_size, shuffle=True
    )
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return trainloader, testloader


def get_partition_label_counts(num_partitions: int) -> dict[int, Counter]:
    """Return {partition_id: Counter({digit: count})} - used only for the
    'show the non-IID split' chart, not by training itself."""
    dataset = _get_federated_dataset(num_partitions)
    counts = {}
    for pid in range(num_partitions):
        labels = dataset.load_partition(pid)["label"]
        counts[pid] = Counter(labels)
    return counts


# -----------------------------------------------------------------------
# Local training / evaluation loops (plain PyTorch, nothing Flower-specific)
# -----------------------------------------------------------------------
def train(model: nn.Module, trainloader: DataLoader, epochs: int, lr: float, device):
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    total_loss, num_batches = 0.0, 0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


def test(model: nn.Module, testloader: DataLoader, device):
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    correct, total_loss, total = 0, 0.0, 0
    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / max(total, 1)
    avg_loss = total_loss / max(len(testloader), 1)
    return avg_loss, accuracy
