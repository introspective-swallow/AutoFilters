import torch
from torch import nn

# Initialize data
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    if isinstance(m, nn.Conv2d):
        nn.init.dirac_(m.weight)


class MLP(nn.Module):
    """ Simple MLP model

    """
    def __init__(self, architecture, output_dim):
        super().__init__()
        self.architecture = architecture
        self.net = self.make_mlp(architecture, output_dim)

    def make_mlp(self, architecture, output_dim):
        layers = []

        layers.append(nn.Flatten())
        for i, hidden_dim in enumerate(architecture):
            layers.append(nn.LazyLinear(hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.LazyLinear(output_dim))
        return nn.Sequential(*layers)
    
    def forward(self, X):
        return self.net(X)
    
    def init_weights(self, dummy_input):
        self.forward(dummy_input)
        self.net.apply(init_weights)
