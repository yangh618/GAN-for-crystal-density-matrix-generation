#!/usr/bin/env python3

import os

import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from constants import *
from GAN import *

# Load density matrices from path, and prepare a dataset
source_file_dir = os.path.dirname(os.path.abspath(__file__))
data_file_path = os.path.join(source_file_dir, f'../dat/dmats.npy')
dmats = torch.tensor(np.load(data_file_path)).to(torch.float)
dmats = dmats.view(-1, 1, NGRID, NGRID, NGRID)
train_set = TensorDataset(dmats)
data_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

D = Discriminator().to(DEVICE)
G = Generator().to(DEVICE)

#Loss function
adversarial_loss  = nn.BCELoss()

#Optimizer
D_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
G_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))

step = 0

D_loss_history = []
G_loss_history = []

D_labels = torch.ones(BATCH_SIZE, 1).to(DEVICE) # Discriminator Label to real
D_fakes = torch.zeros(BATCH_SIZE, 1).to(DEVICE) # Discriminator Label to fake

is_verbose = False
for epoch in tqdm(range(MAX_EPOCH), desc='GAN training'):
    for idx, real_images, in enumerate(data_loader):
        # Training Discriminator
        x = real_images[0].to(DEVICE)
        x_outputs = D(x)
        D_x_loss = adversarial_loss(x_outputs, D_labels)

        # noise vector sampled from a normal distribution
        z = torch.randn((BATCH_SIZE, LATENT_SIZE), device=DEVICE)
        z_outputs = D(G(z))
        D_z_loss = adversarial_loss(z_outputs, D_fakes)
        D_loss = D_x_loss + D_z_loss

        D.zero_grad()
        D_loss.backward()
        D_optimizer.step()

        if step % n_critic == 0:
            # Training Generator
            z = torch.randn(BATCH_SIZE, LATENT_SIZE).to(DEVICE)
            z_outputs = D(G(z))
            G_loss = adversarial_loss(z_outputs, D_labels)

            G.zero_grad()
            G_loss.backward()
            G_optimizer.step()

        if is_verbose:
            if step % 500 == 0:
                print('Epoch: {}/{}, Step: {}, D Loss: {}, G Loss: {}'.format(epoch, MAX_EPOCH, step, D_loss.item(), G_loss.item()))

        step += 1

    D_loss_history.append(D_loss.item())
    G_loss_history.append(G_loss.item())

    torch.save(G.state_dict(), source_file_dir + '/training_models/Generator/G_epoch_%d.pth' % (epoch))
    torch.save(D.state_dict(), source_file_dir + '/training_models/Discriminator/D_epoch_%d.pth' % (epoch))
