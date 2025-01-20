import torch
from torch import nn
import numpy as np
from src.models.base import MLP

# Make initializer
def init_weights(module):  #@save
    """Initialize weights for sequence-to-sequence learning."""
    if type(module) == nn.Linear:
         nn.init.xavier_uniform_(module.weight)

# Make encoder
class MLPEncoder(nn.Module):  #@save
    """The RNN encoder for sequence-to-sequence learning."""
    def __init__(self, architecture, output_dim):
        super().__init__()
        self.net = MLP(architecture, output_dim)

    def forward(self, X):
        return self.net(X)

    

# Make decoder
class MLPDecoder(nn.Module):
    """The RNN decoder for sequence to sequence learning."""
    def __init__(self, architecture, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.net = MLP(architecture, np.prod(output_dim))

    def forward(self, X):
        output = self.net(X)
        #reconstructed = output.reshape(-1, *self.output_dim)
        return output


# Make autoencoder
class MLPAutoencoder(nn.Module):  #@save
    """The RNN encoder--decoder for sequence to sequence learning."""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.net = nn.Sequential(self.encoder, self.decoder)

    def forward(self, X):
        return self.net(X)

    def init_weights(self, dummy_input):
        self.forward(dummy_input)
        self.net.apply(init_weights)