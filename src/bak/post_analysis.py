#!/usr/bin/env python3

import os

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter, CifParser
import tqdm

import matplotlib.pyplot as plt

from constants import *

def check_atomic_number(dmats):

    natoms = np.sum(dmats, (1,2,3))
    natom_min, natom_max = natoms.min(), natoms.max()
    counts, bins = np.histogram(natoms, bins=500, range=(natom_min, natom_max))
    plt.figure()
    plt.stairs(counts, bins)
    plt.xlabel('# of atoms')
    plt.ylabel('# of strcutres')
    plt.title('Atomic number Check')

def check_boundary_conituity(dmats):

    x_max = np.max(dmats, (1,2,3))

    dx_max = np.max(np.abs(dmats[:, 0, :, :] - dmats[:, -1, :, :]), (1,2))
    dy_max = np.max(np.abs(dmats[:, :, 0, :] - dmats[:, :, -1, :]), (1,2))
    dz_max = np.max(np.abs(dmats[:, :, :, 0] - dmats[:, :, :, -1]), (1,2))

    counts, bins = np.histogram(dx_max, bins=500, range=(0, dx_max.max()))
    plt.figure()
    plt.stairs(counts, bins)
    plt.xlabel('dx')
    plt.ylabel('# of strcutres')
    plt.title('Boundary Conituity Check')
    
if __name__ == '__main__':

    # True density matrice
    source_file_dir = os.path.dirname(os.path.abspath(__file__))
    data_file_path = os.path.join(source_file_dir, f'../dat/dmats.npy')
    dmats_real = np.load(data_file_path)
    
    # fake density matrice
    data_file_path = 'fake_dmats.npy'
    dmats_fake = np.load(data_file_path)
    
    check_atomic_number(dmats_real)
    check_atomic_number(dmats_fake)

    check_boundary_conituity(dmats_real)
    check_boundary_conituity(dmats_fake)
    
    plt.show()
