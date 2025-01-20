# Get MNIST dataset using pytorch
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from src.metrics.accuracy import accuracy

DATA = "/home/gui/Repos/AutoFilters/data"

# Load the dataset
transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        ])

train_kwargs = {'batch_size': 64,}
test_kwargs = {'batch_size': 1000}

dataset1 = datasets.MNIST(DATA, train=True, download=True,
                    transform=transform)
dataset2 = datasets.MNIST(DATA, train=False,
                    transform=transform)
train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

from src.models.base import MLP
from src.trainers.base_trainer import train_for_one_epoch

model = MLP(architecture=[64, 32, 16], output_dim=10)
model.init_weights(torch.zeros(64, 1, 28, 28))

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for i in range(5):
    train_loss, valid_loss, valid_acc, _, __ = train_for_one_epoch(
        model,
        optimizer,
        torch.nn.functional.cross_entropy,
        train_loader,
        test_loader,
    )
    print("Train loss: ", train_loss)
    print("Test loss:", valid_loss)
    print("Test accuracy:", valid_acc)
