#!/usr/bin/env python

import numpy as np
import pickle

from pymor.basic import *
import os
from pathlib import Path

from experiment_setup import WaveExperiment, WaveExperimentConfig


def compute_pod_basis():
 
    # Configure experiment
    config = WaveExperimentConfig(x_flow=True, nt=500, timestep_factor=1)
    experiment = WaveExperiment(config)
    modes = 50
    centered = False
   
    timestep_factor = config.timestep_factor
    Nx = config.Nx
    Ny = config.Ny

    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir = Path(script_dir)
    filepaths = experiment.get_filepath_patterns(script_dir)
    
    # base_dir = os.path.join(script_dir, "snapshots_grid")
    # os.makedirs(base_dir, exist_ok=True)

    arrays = []
    for mu_val in [0.5, 0.75, 1]:
        mu_val_tag = f"{mu_val:.2f}".replace('.', '')
        filename = filepaths['snapshots'] / f'snapshots_{Nx}x{Ny}_{mu_val_tag}_nt_{config.nt}'
        with open(filename, 'rb') as f:
            arr = pickle.load(f)['snapshots']
        arrays.append(arr)

    data_mat = np.vstack(arrays)
    print('Raw concatenated shape (flat):', data_mat.shape)

    T_total, n_space2 = data_mat.shape
    n_space = n_space2 // 2
    space = NumpyVectorSpace(n_space)

    if not centered:
        for i, mu in enumerate([0.5, 0.75, 1]):
            q0, p0 = experiment._get_initial_condition(mu_val=mu)
            initial_state = np.hstack((q0, p0)).reshape(-1, 1)
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

    #snapshots_np = scaler.scale(snapshots_np)
    q_flat_scaled = snapshots_np[:, 0, :, :].reshape(T_total, Nx*Ny)
    p_flat_scaled = snapshots_np[:, 1, :, :].reshape(T_total, Nx*Ny)
 
    U = experiment.fom.solution_space.empty()

    for i in range(T_total):
        print(i)
        U.append(experiment.fom.operator.source.make_array([space.from_numpy(q_flat_scaled[i, :]), space.from_numpy(p_flat_scaled[i, :])]))

    # compute reduced basis
    reduced_basis, _ = pod(U, modes=modes)
    reduced_basis_all = reduced_basis.to_numpy()

    # save the reduced basis
    # rb_dir = os.path.join(script_dir, "pod_results")
    # rb_dir = Path(rb_dir)
    # os.makedirs(rb_dir, exist_ok=True)

    if centered:
        rb_path = filepaths['pod_results'] / f"reduced_basis_{Nx}x{Ny}_rbsize_{reduced_basis_all.shape[1]}_nt_{config.nt}.npy"
        np.save(rb_path, reduced_basis_all)
    else:
        rb_path = filepaths['pod_results'] / f"reduced_basis_uncentered_{Nx}x{Ny}_rbsize_{reduced_basis_all.shape[1]}_nt_{config.nt}.npy"
        np.save(rb_path, reduced_basis_all)

if __name__ == '__main__':
    compute_pod_basis()    
