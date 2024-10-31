#!/usr/bin/env python3

import os

import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from constants import *
from GAN import *

source_file_dir = os.path.dirname(os.path.abspath(__file__))
saved_model_dir = os.path.join(source_file_dir, f'training_models/Generator/')
loaded_model_path = os.path.join(saved_model_dir, f'G_epoch_199.pth')


G = Generator().to(DEVICE)
G.load_state_dict(torch.load(loaded_model_path))

G.eval()

num_batch = 64

z = torch.randn((BATCH_SIZE * num_batch, LATENT_SIZE), device=DEVICE)
unknown_mat = G(z).view(BATCH_SIZE*max_sample, NGRID, NGRID, NGRID).detach().to('cpu').numpy()
np.save('fake_dmats.npy', unknown_mat)
