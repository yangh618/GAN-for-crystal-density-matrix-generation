import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from constants import *

torch.manual_seed(1)

class Generator(nn.Module):
    '''
    Generator model definition.
    In the Vanilla GAN it is defined as an MLP model with input size equal to noise vector.
    The output size is the same as images we want to generate.
    '''
    def __init__(self, input_size=LATENT_SIZE, output_size=NGRID**3):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, output_size),

            ###########
            # Tanh() or Sigmoid()
            #nn.Tanh()   #The tanh activation at the output layer ensures that the pixel 
                        #values are mapped in line with its own output, i.e., between (-1, 1)
            nn.Sigmoid() 
            )


    def forward(self, noise_vector):
        generated_img = self.model(noise_vector)
        generated_img = generated_img.view(generated_img.size(0), 1, NGRID, NGRID, NGRID)
        return generated_img

class Discriminator(nn.Module):
    '''
    Discriminator model definition as a binary classifier.
    In the Vanilla GAN it is defined as an MLP model with input size equal to
    flattened image size.
    The output size is the 1 (i.e. the probability of a binary problem -> real or fake).
    '''

    def __init__(self, input_size=NGRID**3, num_classes=1):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, num_classes),
            nn.Sigmoid(), #The final layer has the sigmoid activation function, 
                        #which squashes the output value between 0 (fake) and 1 (real).
            )


    def forward(self, image):
        image_flattened = image.view(image.size(0), -1)
        result = self.model(image_flattened)
        return result


