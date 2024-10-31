#!/usr/bin/env python3

import os

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter, CifParser
import tqdm

import matplotlib.markers as mk
import matplotlib.pyplot as plt

from constants import *

def plot_dmat(dmat):
    '''
    3D visualization of a density matrix.
    '''

    # use endpoint false to avoid double counting at the boundary
    xs = np.linspace(0, 1, NGRID, endpoint=False)

    # generate a grid site matrix using xs, ys (xs), zs (xs)
    X, Y, Z = np.meshgrid(xs, xs, xs)
    
    kw = {
        'vmin': dmat.min(),
        'vmax': dmat.max(),
        'levels': np.linspace(dmat.min(), dmat.max(), 10),
    }

    # Create a figure with 3D ax
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection='3d')

    # Plot contour surfaces
    _ = ax.contourf(
        X[:, :, -1], Y[:, :, -1], dmat[:, :, -1],
        zdir='z', offset=Z.max(), **kw
    )
    _ = ax.contourf(
        X[:, :, 0], Y[:, :, 0], dmat[:, :, 0],
        zdir='z', offset=0, **kw
    )
    _ = ax.contourf(
        X[0, :, :], dmat[0, :, :], Z[0, :, :],
        zdir='y', offset=0, **kw
    )
    _ = ax.contourf(
        X[-1, :, :], dmat[-1, :, :], Z[-1, :, :],
        zdir='y', offset=Y.max(), **kw
    )
    _ = ax.contourf(
        dmat[:, 0, :], Y[:, 0, :], Z[:, 0, :],
        zdir='x', offset=0, **kw
    )
    C = ax.contourf(
        dmat[:, -1, :], Y[:, -1, :], Z[:, -1, :],
        zdir='x', offset=X.max(), **kw
    )
    # -


    # Set limits of the plot from coord limits
    xmin, xmax = X.min(), X.max()
    ymin, ymax = Y.min(), Y.max()
    zmin, zmax = Z.min(), Z.max()
    ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])
    
    # Plot edges
    edges_kw = dict(color='0.4', linewidth=1, zorder=1e3)
    ax.plot([xmax, xmax], [ymin, ymax], 0, **edges_kw)
    ax.plot([xmin, xmax], [ymin, ymin], 0, **edges_kw)
    ax.plot([xmax, xmax], [ymin, ymin], [zmin, zmax], **edges_kw)

    # Set labels and zticks
    ax.set(
        xlabel='X',
        ylabel='Y',
        zlabel='Z',
    )

    # Set zoom and angle view
    ax.view_init(40, -30, 0)
    ax.set_box_aspect(None, zoom=0.9)

    # Colorbar
    fig.colorbar(C, ax=ax, fraction=0.02, pad=0.1, label='Name [units]')

    # Show Figure
    plt.show()
    return 0

if __name__ == '__main__':

    # True density matrice
    source_file_dir = os.path.dirname(os.path.abspath(__file__))
    data_file_path = os.path.join(source_file_dir, f'../dat/dmats.npy')

    # fake density matrice
    # data_file_path = 'fake_dmats.npy'

    dmats = np.load(data_file_path)
    ns, _, _, _ = dmats.shape
    struct_id = np.random.randint(ns)
    print("Plotting structure: No. ", struct_id)
    plot_dmat(dmats[struct_id])

