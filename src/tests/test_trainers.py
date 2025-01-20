import numpy as np
import torch

# from deepclean.trainer import ChunkedTimeSeriesDataset, CompositePSDLoss
from src.trainers.base_trainer import train_for_one_epoch

def make_mlp(input_dim, hidden_dims):
    layers = []
    for i, hidden_dim in enumerate(hidden_dims):
        layers.append(torch.nn.Linear(input_dim, hidden_dim)) 
        layers.append(torch.nn.ReLU())
        input_dim = hidden_dim
    layers.append(torch.nn.Linear(input_dim, 1))
    return torch.nn.Sequential(*layers)


def make_hastie(num_samples, batch_size, shuffle):
    X = np.random.randn(num_samples, 10).astype("float32")
    y = ((X**2).sum(axis=1) > 9.34).astype("float32")[:, None]
    return Dataloader(X, y, batch_size, shuffle)


class Dataloader:
    def __init__(self, X, y, batch_size, shuffle):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
        self.batch_size = batch_size
        self.shuffle = shuffle

    @property
    def num_kernels(self):
        return len(self.X)

    def __len__(self):
        return (len(self.X) - 1) // self.batch_size + 1

    def __iter__(self):
        if self.shuffle:
            self.idx = torch.randperm(self.num_kernels)
        else:
            self.idx = torch.arange(self.num_kernels)
        self.i = 0
        return self

    def __next__(self):
        if (self.i + 1) == len(self):
            raise StopIteration

        idx = self.idx[
            self.i * self.batch_size : (self.i + 1) * self.batch_size
        ]
        self.i += 1
        return self.X[idx], self.y[idx]


def test_train_one_epoch_with_hastie():
    mlp = make_mlp(10, [64, 32, 16])
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    train_data = make_hastie(10000, 128, True)
    valid_data = make_hastie(2000, 256, False)

    for i in range(20):
        train_loss, valid_loss, _, __ = train_for_one_epoch(
            mlp,
            optimizer,
            torch.nn.functional.binary_cross_entropy_with_logits,
            train_data,
            valid_data,
        )