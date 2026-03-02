#!/usr/bin/env python

import numpy as np
import pickle

from pymor.basic import *
import os
from pathlib import Path

from experiment_setup import WaveExperiment, WaveExperimentConfig


def test_pod_galerkin():
 
    # Configure experiment
    config = WaveExperimentConfig(Nx=256, Ny=5, x_flow=True)
    experiment = WaveExperiment(config, mu_val=1.0) #dummy
   
    timestep_factor = config.timestep_factor
    Nx = config.Nx
    Ny = config.Ny

    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir = Path(script_dir)
    
    base_dir = os.path.join(script_dir, "snapshots_grid")
    os.makedirs(base_dir, exist_ok=True)

    number_of_snapshots = 3

    arrays = []
    for i in range(number_of_snapshots):
        filename = os.path.join(base_dir, f'snapshots_{Nx}x{Ny}_{number_of_snapshots}_{i}_nt_{config.nt}_every_{timestep_factor}_ts')
        with open(filename, 'rb') as f:
            arr = pickle.load(f)['snapshots']
        arrays.append(arr)

    data_mat = np.vstack(arrays)
    print('Raw concatenated shape (flat):', data_mat.shape)

    T_total, n_space2 = data_mat.shape
    n_space = n_space2 // 2
    space = NumpyVectorSpace(n_space)
    
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
        U.append(experiment.fom.operator.source.make_array([space.from_numpy(q_flat_scaled[i, :]), space.from_numpy(p_flat_scaled[i, :])]))

    proj_errors = []

    # compute reduced basis
    max_modes = 24
    reduced_basis, _ = pod(U, modes=max_modes)
    print("done computing the POD")
    reduced_basis_all = reduced_basis.to_numpy()

    # save the reduced basis
    rb_dir = os.path.join(script_dir, "pod_results")
    rb_dir = Path(rb_dir)
    os.makedirs(rb_dir, exist_ok=True)

    rb_path = rb_dir / f"reduced_basis_{Nx}x{Ny}_rbsize_{reduced_basis_all.shape[1]}_nt_{config.nt}_every_{timestep_factor}_ts.npy"
    np.save(rb_path, reduced_basis_all)

    mu_val = 0.8
    experiment = WaveExperiment(config, mu_val=mu_val) #dummy
    mu_test = experiment.fom.parameters.parse({'mu': mu_val})
    filepaths = experiment.get_filepath_patterns(script_dir)
    filename = filepaths['snapshots'] / "snapshots_256x5_2_1_nt_1000_every_5_ts"
    with open(filename, 'rb') as f:
        arr = pickle.load(f)['snapshots']
    u_test = np.vstack(arr).T

    q0, p0 = experiment._get_initial_condition(mu_val=mu_val)
    initial_state = np.hstack((q0, - mu_val * p0)).reshape(-1, 1)

    # compute projection error for different RB sizes
    amount_of_iters = int(config.nt/timestep_factor)
    errors = np.zeros((amount_of_iters, 1))
    errors_den = np.zeros((amount_of_iters, 1))
    
    for j in [12, 16, 20, max_modes]:
        reduced_basis = reduced_basis_all[:, :j]
       
        for i in range(amount_of_iters):
            sol_rot_unscaled = u_test[:, i]
            sol_rot_unscaled_enc = reduced_basis.T @ sol_rot_unscaled
            sol_rot_unscaled_dec = reduced_basis @ sol_rot_unscaled_enc
            errors[i, 0] = np.linalg.norm(sol_rot_unscaled_dec.reshape(-1,1) - sol_rot_unscaled.reshape(-1,1))**2
            errors_den[i, 0] = np.linalg.norm(u_test[:, i] + initial_state)**2

        proj_errors.append((j, np.sqrt(np.sum(errors, axis=0) / np.sum(errors_den, axis=0))))

    pod_results_dir = os.path.join(script_dir, "pod_results")
    pod_results_dir = Path(pod_results_dir)
    pod_results_file = pod_results_dir / f"proj_error_{Nx}x{Ny}_maxmodes_{max_modes}_mu_{mu_val}_nt_{config.nt}_every_{timestep_factor}_ts.pkl"

    with pod_results_file.open("wb") as f:
        pickle.dump(proj_errors, f)
    
    print(proj_errors)

if __name__ == '__main__':
    test_pod_galerkin()    
