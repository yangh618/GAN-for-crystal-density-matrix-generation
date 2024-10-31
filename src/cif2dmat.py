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
    INPUT:
    atom_sites: an Natomx3 array.
    Examples: a site list of diamond structure, C8.cI16.
    [[0.00000000000000   0.00000000000000   0.67477395826028]
    [0.67477395826028   0.00000000000000   0.00000000000000]
    [0.00000000000000   0.67477395826028   0.00000000000000]
    [0.32522604173972   0.32522604173972   0.32522604173972]
    [0.00000000000000   0.00000000000000   0.32522604173972]
    [0.32522604173972   0.00000000000000   0.00000000000000]
    [0.00000000000000   0.32522604173972   0.00000000000000]
    [0.67477395826028   0.67477395826028   0.67477395826028]]

    OUTPUT:
    A [NGRID, NGRID, NGRID] density matrix.
    '''
    
    # number of atoms in the structure
    natom, _ = atom_sites.shape

    ############################################################
    # This block will be deleted
    ############################################################

    ############################################################
    
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

    fid =  open(data_file_dir + '../dmats_index.txt', 'w') 
        
    for cif_id, cif_name in enumerate(tqdm.tqdm(cif_files)):
        cif_abs_path = data_file_dir + cif_name
        dmat = cif2dmat(cif_abs_path)

        dmats[cif_id] = dmat
        fid.writelines(str(cif_id) + ' ' + cif_name + '\n')
        #break
    fid.close()
    
    dest_file_dir = os.path.join(source_file_dir, f'../dat/dmats.npy')
    np.save(dest_file_dir, dmats)
