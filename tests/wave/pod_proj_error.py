#!/usr/bin/env python

"""Test for 1D Burgers type equation with POD-Galerkin method.

Usage:
    burgers_test_pod_galerkin.py

Options:
    -h, --help   Show this message.
"""

import numpy as np
import pickle

from pymor.basic import *
from torchpdes.pdes.instationary import wave_2D
import os
from pathlib import Path


def test_pod_galerkin():
 
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir = Path(script_dir)
   
    Nx = 256
    Ny = 256
    sig_pre = 0.5
    sig_pre_tag = f"{sig_pre:.2f}".replace('.', '')
    x_flow = True
    
    T = 1
    nt = 1000
    timestep_factor = 5
    dims = (2, Nx, Ny)

    fom = wave_2D(T=T, Nx=Nx, Ny=Ny, nt=nt, sig_pre=sig_pre, x_flow=x_flow)
    
    mu_test_val = 1.25
    mu_test = fom.parameters.parse({'mu': mu_test_val})
    print(f'Solving for test parameter = {mu_test} ... ')
    u_test = fom.solve(mu_test)
    print("done with FOM solve")

    base_dir = os.path.join(script_dir, "snapshots_grid")
    os.makedirs(base_dir, exist_ok=True)

    number_of_snapshots = 3

    arrays = []
    for i in range(number_of_snapshots):
        filename = os.path.join(base_dir, f'snapshots_{Nx}x{Ny}_sigpre_{sig_pre_tag}_{number_of_snapshots}_{i}_nt_{nt}_every_{timestep_factor}_ts')
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
 
    U = fom.solution_space.empty()

    for i in range(T_total):
        U.append(fom.operator.source.make_array([space.from_numpy(q_flat_scaled[i, :]), space.from_numpy(p_flat_scaled[i, :])]))

    proj_errors = []

    # compute reduced basis
    reduced_basis, _ = pod(U, modes=30)
    print("done computing the POD")
    reduced_basis_all = reduced_basis.to_numpy()

    # save the reduced basis
    rb_dir = os.path.join(script_dir, "pod_results")
    rb_dir = Path(rb_dir)
    os.makedirs(rb_dir, exist_ok=True)

    rb_path = rb_dir / f"reduced_basis_{Nx}x{Ny}_sigpre_{sig_pre}_rbsize_{reduced_basis_all.shape[1]}_nt_{nt}_every_{timestep_factor}_ts.npy"
    np.save(rb_path, reduced_basis_all)

    # compute projection error for different RB sizes
    for j in [8, 16, 24]:
        reduced_basis = reduced_basis_all[:, :j]
        amount_of_iters = int(nt/timestep_factor)
        errors = np.zeros((amount_of_iters, 1))
        errors_den = np.zeros((amount_of_iters, 1))
        for i in range(amount_of_iters):
            sol_rot_unscaled = u_test.to_numpy()[:, timestep_factor*i] - u_test.to_numpy()[:, 0]
            sol_rot_unscaled_enc = reduced_basis.T @ sol_rot_unscaled
            sol_rot_unscaled_dec = reduced_basis @ sol_rot_unscaled_enc
            errors[i, 0] = np.linalg.norm(sol_rot_unscaled_dec.reshape(-1,1) - sol_rot_unscaled.reshape(-1,1))**2
            errors_den[i, 0] = np.linalg.norm(u_test.to_numpy()[:, timestep_factor*i])**2

        proj_errors.append((j, np.sqrt(np.sum(errors, axis=0) / np.sum(errors_den, axis=0))))

    pod_results_dir = os.path.join(script_dir, "pod_results")
    pod_results_dir = Path(pod_results_dir)
    pod_results_file = pod_results_dir / f"proj_error_{Nx}x{Ny}_noscale_norot_mu_{mu_test_val}_nt_{nt}_every_{timestep_factor}_ts.pkl"

    with pod_results_file.open("wb") as f:
        pickle.dump(proj_errors, f)
    
    print(proj_errors)

if __name__ == '__main__':
    test_pod_galerkin()    
