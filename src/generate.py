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

max_sample = 100
for sample_id in range(max_sample):
        z = torch.randn((BATCH_SIZE, LATENT_SIZE), device=DEVICE)
        unknown_mat = G(z)

        print(unknown_mat.shape)
        natom = torch.sum(unknown_mat, (1,2,3,4))
        print(natom)
