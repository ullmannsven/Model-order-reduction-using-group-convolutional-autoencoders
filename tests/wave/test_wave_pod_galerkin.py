#!/usr/bin/env python

"""Test for 1D Burgers type equation with Deep-Galerkin method.

Usage:
    burgers_test_deep_galerkin.py P_RED NETWORK_TYPE FILENAME

Arguments:
    P_RED                     Reduced basis size.
    NETWORK_TYPE              Type of the neural network ('simple', 'maxpooling' or 'new_architecture')
    FILENAME                  Name of the file containing the network data.

Options:
    -h, --help   Show this message.
"""

import time
import numpy as np
import pickle
import os
from pathlib import Path

from pymor.basic import *

import torch

from torchpdes.models.instationary.pod_galerkin_utilities_IMR import POD_Galerkin_quasi_newton
from torchpdes.pdes.instationary import wave_2D
from scale import Scaler

def test_wave_2D():

    p_red = 16
    Nx = 51
    Ny = 51
    
    Lx = 1
    Ly = 1
    x_flow = True
    sig_pre = 2
    if x_flow: 
        hx = Lx/(Nx)
        hy = Ly/(Ny-1)
    else: 
        hx = Lx/(Nx-1)
        hy = Ly/Ny

    #load the already computed RB size 
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rb_dir = os.path.join(script_dir, "pod_results")
    script_dir = Path(script_dir)
    rb_dir = Path(rb_dir)
    os.makedirs(rb_dir, exist_ok=True)

    rb_path = rb_dir / f"reduced_basis_{Nx}x{Ny}_sigpre_{sig_pre}_rbsize_30.npy"
    reduced_basis_all = np.load(rb_path)
    reduced_basis = reduced_basis_all[:, :p_red]

    dt = 1 / 200
    T = 1
    nt = int(T / dt)
    dims = (2, Nx, Ny)

    fom = wave_2D(T=T, Nx=Nx, Ny=Ny, sig_pre=sig_pre, x_flow=x_flow)
    scaler = Scaler(dims=dims)
    
    mu_test_val = 1.0
    mu_test = fom.parameters.parse({'mu': mu_test_val})
    print(f'Solving for test parameter = {mu_test} ... ')
    u_test = fom.solve(mu_test)
    print("done with FOM solve")

    #This is not doing anything, just to keep it aligned with the AE version
    zero_vec = np.zeros(2*Nx*Ny)
    u_0_hat = reduced_basis.T @ zero_vec
    decoded_u_0_hat = reduced_basis @ u_0_hat
    
    #Initial condition of the FOM - here we can now rotate compared to the training data
    if x_flow:
        x = np.linspace(0, Lx, Nx, endpoint=False)
        y = np.linspace(0, Ly, Ny, endpoint=True)
        X, _ = np.meshgrid(x, y, indexing='xy')
        x0 = 0.3
        sig = 0.05
        q0_flat = np.exp(-((X - x0)**2) / (sig_pre*sig**2)).ravel()

        Q = q0_flat.reshape(Ny, Nx)
        dqdx_mat = np.zeros_like(Q)
        dqdx_mat[:, 1:-1] = (Q[:, 2:] - Q[:, :-2])/(2*hx)
        dqdx_mat[:, 0] = (Q[:, 1] - Q[:, -1])/(2*hx)
        dqdx_mat[:, -1] = (Q[:, 0] - Q[:, -2])/(2*hx)
        dqdx_flat = dqdx_mat.ravel()
        p0_flat = - mu_test_val * dqdx_flat

    else:
        x = np.linspace(0, Lx, Nx, endpoint=True)
        y = np.linspace(0, Ly, Ny, endpoint=False)
        _, Y = np.meshgrid(x, y, indexing='xy')

        y0 = 0.3
        sig = 0.05
        q0_flat = np.exp(-((Y - y0)**2) / (sig_pre*sig**2)).ravel()

        Q = q0_flat.reshape(Ny, Nx)
        dqdy_mat = np.zeros_like(Q)
        dqdy_mat[1:-1, :] = (Q[2:, :] - Q[:-2, :])/(2*hy)
        dqdy_mat[0, :] = (Q[1, :] - Q[-1, :])/(2*hy)
        dqdy_mat[-1, :] = (Q[0, :] - Q[-2, :])/(2*hy)
        dqdy_flat = dqdy_mat.ravel()
        p0_flat = - mu_test_val * dqdy_flat

    u_ref = np.hstack((q0_flat, p0_flat)).reshape(-1,1) - decoded_u_0_hat.reshape(-1,1)

    u_approx = [u_0_hat.reshape(1,-1)]
    u_approx_full = [np.hstack((q0_flat, p0_flat)).reshape(-1,1)]
    
    # Implicit midpoint timestepping for ROM
    for i in range(nt):
        tic = time.time()
        t = (i + 1) * dt
        print(f'Time: {t}')
        u_n1 = u_approx[-1]

        ## Quasi Newton:
        u_new = POD_Galerkin_quasi_newton(reduced_basis, u_n1, mu_test, dt, fom, u_ref, tol=1e-8)
        ## End Quasi Newton

        u_approx.append(u_new)
        decode_u_new = reduced_basis @ u_new.T
        u_approx_full.append((u_ref + decode_u_new).reshape(-1,1))
        print(f"Step took {time.time()-tic}")

    # save the results
    filename = script_dir / f'approx_full_pod_galerkin_p_{p_red}'
    with open(filename, 'wb') as file_obj:
        pickle.dump({'mu': mu_test, 'u_pod_galerkin': u_approx_full, 'u_full': u_test.to_numpy()}, file_obj)
    
    # compute the error
    error_sum = 0
    error_sum_q = 0
    error_sum_p = 0
    norm_sum = 0
    norm_sum_q = 0
    norm_sum_p = 0
    latent_error = 0
    latent_error_norm = 0
    for i in range(nt+1):
        error_sum += np.linalg.norm(u_approx_full[i] - u_test.to_numpy()[:, i].reshape(-1,1))**2
        norm_sum += np.linalg.norm(u_test.to_numpy()[:, i])**2

        error_sum_q += np.linalg.norm(u_approx_full[i][:Nx*Ny] - u_test.to_numpy()[:Nx*Ny, i].reshape(-1,1))**2
        norm_sum_q += np.linalg.norm(u_test.to_numpy()[:Nx*Ny, i])**2

        error_sum_p += np.linalg.norm(u_approx_full[i][Nx*Ny:] - u_test.to_numpy()[Nx*Ny:, i].reshape(-1,1))**2
        norm_sum_p += np.linalg.norm(u_test.to_numpy()[Nx*Ny:, i])**2

        # sol = torch.as_tensor(scaler.restrict(u_test.to_numpy()[:, i] - u_test.to_numpy()[:, 0]), dtype=torch.double, device="cpu").unsqueeze(0)
        # sol_enc = model.network.encode(sol).detach().cpu().numpy()

        # latent_error_ = np.linalg.norm(sol_enc.reshape(-1,1) - u_approx[i].reshape(-1,1))**2
        # latent_error_norm_ = np.linalg.norm(sol_enc)**2
        # latent_error += latent_error_
        # latent_error_norm += latent_error_norm_

        #print("iteration, latent error", i, np.sqrt(latent_error) / np.sqrt(latent_error_norm))
        #print("iteration, reconstruction error", i, np.sqrt(error_sum) / np.sqrt(norm_sum))
        #print()

        # if i == nt: 
        #     space = NumpyVectorSpace(model.dims[0]*model.dims[1]*model.dims[2])
        #     fom.visualize(space.from_numpy(u_approx_full[i] - u_test.to_numpy()[:, i].reshape(-1,1)))

    relative_error = np.sqrt(error_sum) / np.sqrt(norm_sum)
    print(f'Relative error total: {relative_error}')

    relative_error_q = np.sqrt(error_sum_q) / np.sqrt(norm_sum_q)
    print(f'Relative error q: {relative_error_q}')

    relative_error_p = np.sqrt(error_sum_p) / np.sqrt(norm_sum_p)
    print(f'Relative error p: {relative_error_p}')

    #relative_error_latent = np.sqrt(latent_error) / np.sqrt(latent_error_norm)
    #print(f'Relative error latent: {relative_error_latent}')

    relative_error_file = script_dir / "test_relative_errors_wave.txt"
    with open(relative_error_file, "a") as f:
        f.write(f'POD-Galerkin\t{p_red}\t{mu_test_val}\t{relative_error}\n')

if __name__ == '__main__':
    test_wave_2D()
