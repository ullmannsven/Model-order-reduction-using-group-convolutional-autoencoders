#!/usr/bin/env python

import numpy as np
import pickle
import os
from pathlib import Path

from pymor.basic import *
from pymor.algorithms.symplectic import psd_cotangent_lift
from experiment_setup import WaveExperimentConfig, WaveExperiment

def compute_cl_basis(): 

    # Configure experiment
    config = WaveExperimentConfig(nt=500, timestep_factor=1)
    experiment = WaveExperiment(config) #dummy
    max_modes = 50
    centered = False

    Nx = config.Nx
    Ny = config.Ny

    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir = Path(script_dir)
    filepaths = experiment.get_filepath_patterns(script_dir)

    arrays = []
    for mu_val in [0.5, 0.75, 1]:
        mu_tag =  f"{mu_val:.2f}".replace('.', '')
        filename = os.path.join(filepaths['snapshots'], f'snapshots_{Nx}x{Ny}_{mu_tag}_nt_{config.nt}')
        with open(filename, 'rb') as f:
            arr = pickle.load(f)['snapshots']
        arrays.append(arr)

    data_mat = np.vstack(arrays)
    print('Raw concatenated shape (flat):', data_mat.shape)

    T_total, n_space2 = data_mat.shape
    n_space = n_space2 // 2
    space = NumpyVectorSpace(n_space)

    if not centered:
        for i, mu_val in enumerate([0.5, 0.75, 1]):
            initial_state = experiment.get_initial_state(mu_val=mu_val)
            data_mat[i*config.nt:(i+1)*config.nt, :] = data_mat[i*config.nt:(i+1)*config.nt, :] + initial_state.T
    
    # Slice q and p blocks
    q_flat = data_mat[:, :n_space]      
    p_flat = data_mat[:, n_space:]

    # Reshape each time slice to images and stack as (T, 2, Ny, Nx)
    snapshots_np = np.empty((T_total, 2, Ny, Nx), dtype=np.float64)
    for t in range(T_total):
        q_img = q_flat[t, :].reshape(Ny, Nx)
        p_img = p_flat[t, :].reshape(Ny, Nx)
        snapshots_np[t, 0, :, :] = q_img
        snapshots_np[t, 1, :, :] = p_img

    q_flat_scaled = snapshots_np[:, 0, :, :].reshape(T_total, Nx*Ny)
    p_flat_scaled = snapshots_np[:, 1, :, :].reshape(T_total, Nx*Ny)

    # compute reduced basis
    U = experiment.fom.operator.source.make_array([space.from_numpy(q_flat_scaled.T), space.from_numpy(p_flat_scaled.T)])
    reduced_basis = psd_cotangent_lift(U, modes=max_modes)

    # save the CL basis
    if centered:
        rb_path = filepaths['cl_results'] / f"reduced_basis_{Nx}x{Ny}_rbsize_{max_modes}_nt_{config.nt}"
    else: 
        rb_path = filepaths['cl_results'] / f"reduced_basis_uncentered_{Nx}x{Ny}_rbsize_{max_modes}_nt_{config.nt}"
    with open(rb_path, 'wb') as file:
        pickle.dump(reduced_basis, file)

if __name__ == '__main__':
    compute_cl_basis()
