from src.tests.test_trainers import test_train_one_epoch_with_hastie
from src.models.autoencoder import Encoder, Decoder, Autoencoder
from src.trainers.base_trainer import train_for_one_epoch

from src.datasets.utils import MTFraEng
import logging
import torch
from torch import nn

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

data = MTFraEng(root='/home/gui/Repos/AutoFilters/data',batch_size=128)
train_data = data.get_dataloader(train=True)

for (X, y) in train_data:
    print(X.shape, y.shape)
    break
valid_data = data.get_dataloader(train=False)

embed_size, num_hiddens, num_layers, dropout = 256, 256, 2, 0.2
encoder = Encoder(
    len(data.src_vocab), embed_size, num_hiddens, num_layers, dropout)
decoder = Decoder(
    len(data.tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
model = Autoencoder(encoder, decoder)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

loss = nn.CrossEntropyLoss()

def criterion(Y_hat, Y, tgt_pad=data.tgt_vocab['<pad>']):
    l = loss(Y_hat, Y, averaged=False)
    mask = (Y.reshape(-1) != tgt_pad).type(torch.float32)
    return (l * mask).sum() / mask.sum()

for i in range(20):
    train_loss, valid_loss, _, __ = train_for_one_epoch(
        model,
        optimizer,
        criterion,
        train_data,
        valid_data,
    )

    
    if valid_loss > 0.9:
        break
else:
    raise ValueError(
        f"Couldn't converge, valid loss is {valid_loss}"
    )