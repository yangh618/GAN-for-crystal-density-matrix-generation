#!/usr/bin/env python3

import os

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter, CifParser
import tqdm

from constants import *

def site2dmat(atom_sites: np.ndarray) -> np.ndarray:
    '''
    Convert a site-list representation of a crystal structure "atom_sites" to a density matrix representation "dmat", or number density matrix if number is used.
    '''
    # number of atoms in the structure
    natom, _ = atom_sites.shape

    # use endpoint false to avoid double counting at the boundary
    xs = np.linspace(0, 1, NGRID, endpoint=False)

    # generate a grid site matrix using xs, ys (xs), zs (xs)
    grid_sites = np.array(np.meshgrid(xs, xs, xs))

    # tranpose atom_sites to make it compatible with grid_sites
    atom_sites = atom_sites.T

    # expand dimensions of grid_sites and atom_sites to make compatible shape
    grid_sites = grid_sites[:, np.newaxis,:, :]
    atom_sites = atom_sites[:, :, np.newaxis, np.newaxis, np.newaxis]

    # Use this trick to solve the boundary contidition
    distances = (grid_sites - atom_sites + 0.5) % 1 - 0.5
    distances = np.sum(distances ** 2, axis=0)

    # Compute the number density matrix
    dmat = np.exp(-distances/2/SIGMA**2) /SIGMA**3 /np.sqrt(2*PI)**3
    dmat = np.sum(dmat, 0) / NGRID**3

    # Uncomment the following line to check your tensor operations
    # print(atom_sites.shape, grid_sites.shape, distances.shape, dmat.shape)

    # For a correct number density matrix, its sum will be the number of atoms in the structure
    assert np.abs(np.sum(dmat) - natom) < 1e-3, "Incorrect density matrix!!!"

    return dmat
          
def cif2dmat(cif_name: str) -> np.ndarray:
    '''
    Read a structure from a cif file "cif_name" and convert it to the density matrix.
    '''

    # Parse structure from a cif file
    try:
        parser = CifParser(cif_name)
    except :
        print('File does not exist!')
        return 0

    struct = parser.parse_structures()[0]
    fcoords = struct.frac_coords % 1 # use %1 to rescale to [0, 1)

    dmat = site2dmat(fcoords)

    return dmat


if __name__ == '__main__':

    # define directories of source files and data files
    source_file_dir = os.path.dirname(os.path.abspath(__file__))
    data_file_dir = os.path.join(source_file_dir, f'../dat/cif/')

    cif_files = os.listdir(data_file_dir)
    cif_num = len(cif_files)
    
    dmats = np.zeros((cif_num, NGRID, NGRID, NGRID))
    
    for cif_id, cif_name in enumerate(tqdm.tqdm(cif_files)):
        cif_abs_path = data_file_dir + cif_name
        dmat = cif2dmat(cif_abs_path)

        dmats[cif_id] = dmat
        #break

    dest_file_dir = os.path.join(source_file_dir, f'../dat/dmats.npy')
    np.save(dest_file_dir, dmats)
