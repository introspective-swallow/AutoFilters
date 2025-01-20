# Get MNIST dataset using pytorch
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from src.metrics.accuracy import accuracy
from src.models.autoencoder import MLPEncoder, MLPDecoder, MLPAutoencoder
from src.models.base import MLP
from src.trainers.base_trainer import train_autoencoder_for_one_epoch
import wandb
import random

learning_rate = 0.001
epochs = 10

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="AutoFilter",

    # track hyperparameters and run metadata
    config={
    "learning_rate": learning_rate,
    "architecture": "MLP-Autoencoder",
    "dataset": "MNIST",
    "epochs": epochs,
    }
)



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

encoder = MLPEncoder(architecture=[64, 32, 16], output_dim=10)
decoder = MLPDecoder(architecture=[16, 32, 64], output_dim=28*28)
model = MLPAutoencoder(encoder, decoder)
model.init_weights(torch.zeros(64, 1, 28, 28))


loss = torch.nn.functional.mse_loss

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for i in range(epochs):
    train_loss, valid_loss, _, __ = train_autoencoder_for_one_epoch(
        model,
        optimizer,
        loss,
        train_loader,
        test_loader,
    )
    print("Train loss: ", train_loss)
    print("Test loss:", valid_loss)

    # log metrics to wandb
    wandb.log({"Train loss": train_loss, "Valid loss": valid_loss})


# Plot reconstructed images
import matplotlib.pyplot as plt
import numpy as np

model.eval()

for i, (X, y) in enumerate(test_loader):
    X_hat = model(X)
    break

fig, axs = plt.subplots(2, 5, figsize=(10, 5))
for i in range(5):
    axs[0, i].imshow(X[i].squeeze().numpy())
    axs[1, i].imshow(X_hat[i].detach().reshape(28,28).numpy())

plt.show()