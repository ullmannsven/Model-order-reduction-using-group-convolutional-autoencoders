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
    experiment = WaveExperiment(config, mu_val=100) #dummy
    max_modes = 50

    Nx = config.Nx
    Ny = config.Ny

    script_dir = os.path.dirname(os.path.abspath(__file__))
    rb_dir = os.path.join(script_dir, "CL_results")
    script_dir = Path(script_dir)
    rb_dir = Path(rb_dir)
    os.makedirs(rb_dir, exist_ok=True)

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

    for i, mu in enumerate([0.5, 0.75, 1]):
        q0, p0 = experiment._get_initial_condition(mu_val=mu)
        initial_state = np.hstack((q0, - mu * p0)).reshape(-1, 1)
        data_mat[i*200:(i+1)*200, :] = data_mat[i*200:(i+1)*200, :] + initial_state.T
    
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
    reduced_basis = psd_cotangent_lift(U, modes=max_modes)

    # save the CL basis
    rb_path = rb_dir / f"reduced_basis_uncentered_CL_{Nx}x{Ny}_rbsize_{max_modes}_nt_{config.nt}"
    with open(rb_path, 'wb') as file:
        pickle.dump(reduced_basis, file)

if __name__ == '__main__':
    compute_cl_basis()
