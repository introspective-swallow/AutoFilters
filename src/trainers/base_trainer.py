import logging
import os
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from src.metrics.accuracy import accuracy

def train_autoencoder_for_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    train_data: None,
    valid_data: None,
    profiler: Optional[torch.profiler.profile] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
):
    """Run a single epoch of training"""

    train_loss = 0
    samples_seen = 0
    start_time = time.time()
    model.train()
    for (X, y) in tqdm(train_data):
        optimizer.zero_grad(set_to_none=True)  # reset gradient
        # do forward step in mixed precision
        # if a gradient scaler got passed
        with torch.autocast("cuda", enabled=scaler is not None):
            prediction = model(X)
            loss = criterion(prediction, X.reshape(-1, np.prod(X.shape[1:])))

        train_loss += loss.item() * len(X)
        samples_seen += len(X)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if profiler is not None:
            profiler.step()
        logging.debug(f"{samples_seen}")

    if profiler is not None:
        profiler.stop()

    end_time = time.time()
    duration = end_time - start_time
    throughput = samples_seen / duration
    train_loss /= samples_seen
    
    logging.info(
        "Duration {:0.2f}s, Throughput {:0.1f} samples/s".format(
            duration, throughput
        )
    )
    msg = f"Train Loss: {train_loss:.4e}"

    # Evaluate performance on validation set if given
    if valid_data is not None:
        valid_loss = 0
        valid_acc = 0
        samples_seen = 0

        model.eval()
        with torch.no_grad():
            for (X, y) in valid_data:
                prediction = model(X)
                loss = criterion(prediction, X.reshape(-1, np.prod(X.shape[1:])))
                valid_loss += loss.item() * len(X)
                samples_seen += len(X)
        valid_loss /= samples_seen
        msg += f", Valid Loss: {valid_loss:.4e}"
    else:
        valid_loss = None

    logging.info(msg)
    return train_loss, valid_loss, duration, throughput

def train_classifier_for_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    train_data: None,
    valid_data: None,
    profiler: Optional[torch.profiler.profile] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
):
    """Run a single epoch of training"""

    train_loss = 0
    samples_seen = 0
    start_time = time.time()
    model.train()
    for (X, y) in tqdm(train_data):
        optimizer.zero_grad(set_to_none=True)  # reset gradient
        # do forward step in mixed precision
        # if a gradient scaler got passed
        with torch.autocast("cuda", enabled=scaler is not None):
            prediction = model(X)
            loss = criterion(prediction, y)

        train_loss += loss.item() * len(X)
        samples_seen += len(X)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if profiler is not None:
            profiler.step()
        logging.debug(f"{samples_seen}")

    if profiler is not None:
        profiler.stop()

    end_time = time.time()
    duration = end_time - start_time
    throughput = samples_seen / duration
    train_loss /= samples_seen
    
    logging.info(
        "Duration {:0.2f}s, Throughput {:0.1f} samples/s".format(
            duration, throughput
        )
    )
    msg = f"Train Loss: {train_loss:.4e}"

    # Evaluate performance on validation set if given
    if valid_data is not None:
        valid_loss = 0
        valid_acc = 0
        samples_seen = 0

        model.eval()
        with torch.no_grad():
            for (X, y) in valid_data:
                prediction = model(X)
                loss = criterion(prediction, y)

                valid_loss += loss.item() * len(X)
                valid_acc += accuracy(prediction, y)
                samples_seen += len(X)

        valid_loss /= samples_seen
        valid_acc /= len(valid_data) # number of batches
        msg += f", Valid Loss: {valid_loss:.4e}, Valid Acc: {valid_acc*100:.4f}%"
    else:
        valid_loss = None
        valid_acc = None

    logging.info(msg)
    return train_loss, valid_loss, valid_acc, duration, throughput
