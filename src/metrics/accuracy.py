import torch

def accuracy(preds, labels):
    """
    Compute the accuracy of a model on a dataset
    """
    # Get the index of the max log-probability
    preds = torch.argmax(preds, dim=1)
    return (preds == labels).float().mean()