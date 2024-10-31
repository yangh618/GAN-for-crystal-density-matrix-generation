#!/usr/bin/env python3

import torch

# Mathematical constants
PI = 3.1415926


# Some Predefined constants for density matrix
NGRID = 16
SIGMA = 0.1


# Predefined constants for GAN model

LATENT_SIZE = 20

# training setup
BATCH_SIZE = 64
MAX_EPOCH = 200
n_critic = 1 # for training more k steps about Discriminator

# specify device
try:
    DEVICE
except NameError:
    # if DEVICE is not defined, define it
    print("GPU Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda:0")
    else:
        DEVICE = "cpu"
    print(f"Using {DEVICE} for training.")

    
