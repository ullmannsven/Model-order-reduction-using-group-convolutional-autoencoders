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

from torchpdes.neuralnetworks.autoencoders import GCNNAutoencoder2D, RotationGCNNAutoencoder2D, CNNAutoencoder2D
from torchpdes.models.instationary.nonlinear_manifolds import NonlinearManifoldsMOR2D
from torchpdes.models.instationary.deep_galerkin_utilities_IMR import Galerkin_quasi_newton
from torchpdes.models.instationary.deep_lspg_utilities_IMR import LSPG_quasi_newton
from torchpdes.pdes.instationary import wave_2D
from scale import Scaler

def test_wave_2D():

    p_red = 16
    Nx = 256
    Ny = 256
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir = Path(script_dir)

    nn_save_filepath = os.path.join(script_dir, "checkpoints")
    nn_save_filepath = Path(nn_save_filepath) / "wave_2D_CNN_256x256_p_16_t_31_10_2025-14_40_31_weighted_scaling_circular_padding_onecycle.pt"

    network_parameters_dir = os.path.join(script_dir, "network_parameters") 
    network_parameters_file = Path(network_parameters_dir) / "wave_2D_CNN_256x256_p_16_t_31_10_2025-14_40_31_weighted_scaling_circular_padding_onecycle.pkl"

    with Path(network_parameters_file).open("rb") as f:
        parameters = pickle.load(f)

    assert f"_{Nx}x{Ny}_" in str(nn_save_filepath)
    assert f"p_{p_red}_" in str(nn_save_filepath)

    Lx = 1
    Ly = 1
    x_flow = True
    T = 1
    nt = 1000
    sig_pre = 0.5
    timestep_factor = 5

    if x_flow: 
        hx = Lx/(Nx)
        hy = Ly/(Ny-1)
    else: 
        hx = Lx/(Nx-1)
        hy = Ly/Ny

    dims = (2, Nx, Ny)

    fom = wave_2D(T=T, Nx=Nx, Ny=Ny, sig_pre=sig_pre, nt=nt, x_flow=x_flow)
    scaler = Scaler(dims=dims)
    
    model = NonlinearManifoldsMOR2D(network=CNNAutoencoder2D,
                                    scaler=scaler,
                                    dims=dims,
                                    network_parameters=parameters['network_parameters']
                                    )

    model.load_neural_network(path=nn_save_filepath)
    model.network.eval()

    trainable = sum(p.numel() for p in model.network.parameters() if p.requires_grad)
    print("so viele parameter hat mein netz", trainable)

    mu_test_val = 1.0
    mu_test = fom.parameters.parse({'mu': mu_test_val})

    # TODO: Computing a new solution takes very long. Thats we we for now just test on training data
    snapshots_grids_dir = os.path.join(script_dir, "snapshots_grid")
    filename = Path(snapshots_grids_dir) / f"snapshots_256x256_sigpre_050_3_1_nt_1000_every_5_ts"
    with open(filename, 'rb') as f:
        arr = pickle.load(f)['snapshots']

    u_test = np.vstack(arr).T

    #print(f'Solving for test parameter = {mu_test} ... ')
    #u_test = fom.solve(mu_test)
    #print("done with FOM solve")

    zero_vec = np.zeros(2*Nx*Ny)
    #zero_mat = torch.as_tensor(scaler.scale(scaler.restrict(zero_vec)), dtype=torch.double, device="cpu").unsqueeze(0)
    zero_mat = torch.as_tensor(scaler.restrict(zero_vec), dtype=torch.double, device="cpu").unsqueeze(0)
    u_0_hat = model.network.encode(zero_mat).detach().cpu().numpy()
    decoded_u_0_hat = model.network.decode(torch.as_tensor(u_0_hat, dtype=torch.double, device="cpu"))[0, :, :, :].detach().cpu().numpy()
    #decoded_u_0_hat = scaler.prolongate(scaler.unscale(decoded_u_0_hat))
    decoded_u_0_hat = scaler.prolongate(decoded_u_0_hat)
    
        
    #Initial condition of the FOM - here we can now rotate compared to the training data
    if x_flow:
        x = np.linspace(0, Lx, Nx, endpoint=False)
        y = np.linspace(0, Ly, Ny, endpoint=True)
        X, _ = np.meshgrid(x, y, indexing='xy')
        x0 = 0.3
        sig = 0.05
        q0_flat = np.exp(-((X - x0)**2) / (2*sig_pre*sig**2)).ravel()

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
        q0_flat = np.exp(-((Y - y0)**2) / (2*sig_pre*sig**2)).ravel()

        Q = q0_flat.reshape(Ny, Nx)
        dqdy_mat = np.zeros_like(Q)
        dqdy_mat[1:-1, :] = (Q[2:, :] - Q[:-2, :])/(2*hy)
        dqdy_mat[0, :] = (Q[1, :] - Q[-1, :])/(2*hy)
        dqdy_mat[-1, :] = (Q[0, :] - Q[-2, :])/(2*hy)
        dqdy_flat = dqdy_mat.ravel()
        p0_flat = - mu_test_val * dqdy_flat

    u_ref = np.hstack((q0_flat, p0_flat)).reshape(-1,1) - decoded_u_0_hat

    u_approx = [u_0_hat]
    u_approx_full = [np.hstack((q0_flat, p0_flat)).reshape(-1,1)]
    
    # Implicit midpoint timestepping for ROM

    for i in range(int(nt/timestep_factor)):
        tic = time.time()
        dt = timestep_factor / nt
        t = (i + 1) * dt
        print(f'Time: {t}')
        u_n1 = u_approx[-1]

        ## Quasi Newton:
        u_new = LSPG_quasi_newton(model, u_n1, mu_test, dt, fom, u_ref, tol=1e-6)
        ## End Quasi Newton

        u_approx.append(u_new)

        decode_u_new = model.network.decode(torch.as_tensor(u_new, dtype=torch.double, device="cpu"))[0].detach().cpu().numpy()
        #decode_u_new = scaler.prolongate(scaler.unscale(decode_u_new))
        decode_u_new = scaler.prolongate(decode_u_new)

        u_approx_full.append((u_ref + decode_u_new).reshape(-1,1))
        print(f"Step took {time.time()-tic}")

    # for i in range(int(nt/timestep_factor)):
    #     tic = time.time()
    #     dt = timestep_factor / nt
    #     t = (i + 1) * dt
    #     print(f'Time: {t}')
    #     u_n1 = u_approx[-1]

    #     ## Quasi Newton
    #     u_new = LSPG_quasi_newton(model, u_n1, mu_test, dt, fom, u_ref, tol=1e-6)
        
    #     ## Project back to encoder manifold
    #     decoded_u_new = model.network.decode(torch.as_tensor(u_new, dtype=torch.double, device="cpu"))[0].detach().cpu().numpy()
    #     decoded_u_new = model.scaler.prolongate(decoded_u_new)
    #     full_state_new = u_ref + decoded_u_new
        
    #     # Re-encode to get the "correct" latent representation
    #     with torch.no_grad():
    #         full_state_tensor = torch.as_tensor(scaler.restrict(full_state_new.reshape(-1,1) - u_approx_full[0].reshape(-1,1)), dtype=torch.double, device="cpu").unsqueeze(0)
    #         u_new_projected = model.network.encode(full_state_tensor).detach().cpu().numpy()
        
    #     # Check how much we had to correct
    #     correction = np.linalg.norm(u_new_projected - u_new)
    #     print(f" Latent correction: {correction:.6e}")
        
    #     u_approx.append(u_new_projected)  # Use projected version
        
    #     # Decode the projected version for full-order state
    #     decode_u_new = model.network.decode(torch.as_tensor(u_new_projected, dtype=torch.double, device="cpu"))[0, :, :, :].detach().cpu().numpy()
    #     decode_u_new = scaler.prolongate(decode_u_new)
        
    #     u_approx_full.append((u_ref + decode_u_new).reshape(-1,1))
    #     print(f"Step took {time.time()-tic}")

    # save the results
    mor_results_filepath = os.path.join(script_dir, "mor_results")
    filename = Path(mor_results_filepath) / f'approx_full_deep_galerkin_p_{p_red}'
    with open(filename, 'wb') as file_obj:
        pickle.dump({'mu': mu_test, 'u_deep_galerkin': u_approx_full, 'u_full': u_test}, file_obj)
    
    # compute the error
    error_sum = 0
    error_sum_q = 0
    error_sum_p = 0
    norm_sum = 0
    norm_sum_q = 0
    norm_sum_p = 0
    latent_error = 0
    latent_error_norm = 0
    for i in range(int(nt/timestep_factor)):
        # error_sum += np.linalg.norm(u_approx_full[i] - u_test.to_numpy()[:, i].reshape(-1,1))**2
        # norm_sum += np.linalg.norm(u_test.to_numpy()[:, i])**2

        # error_sum_q += np.linalg.norm(u_approx_full[i][:Nx*Ny] - u_test.to_numpy()[:Nx*Ny, i].reshape(-1,1))**2
        # norm_sum_q += np.linalg.norm(u_test.to_numpy()[:Nx*Ny, i])**2

        # error_sum_p += np.linalg.norm(u_approx_full[i][Nx*Ny:] - u_test.to_numpy()[Nx*Ny:, i].reshape(-1,1))**2
        # norm_sum_p += np.linalg.norm(u_test.to_numpy()[Nx*Ny:, i])**2

        # sol = torch.as_tensor(scaler.restrict(u_test.to_numpy()[:, i] - u_test.to_numpy()[:, 0]), dtype=torch.double, device="cpu").unsqueeze(0)
        # sol_enc = model.network.encode(sol).detach().cpu().numpy()

        error_sum += np.linalg.norm(u_approx_full[i] - u_test[:, i].reshape(-1,1))**2
        norm_sum += np.linalg.norm(u_test[:, i])**2

        error_sum_q += np.linalg.norm(u_approx_full[i][:Nx*Ny] - u_test[:Nx*Ny, i].reshape(-1,1))**2
        norm_sum_q += np.linalg.norm(u_test[:Nx*Ny, i])**2

        error_sum_p += np.linalg.norm(u_approx_full[i][Nx*Ny:] - u_test[Nx*Ny:, i].reshape(-1,1))**2
        norm_sum_p += np.linalg.norm(u_test[Nx*Ny:, i])**2

        sol = torch.as_tensor(scaler.restrict(u_test[:, i] - u_test[:, 0]), dtype=torch.double, device="cpu").unsqueeze(0)
        sol_enc = model.network.encode(sol).detach().cpu().numpy()

        latent_error_ = np.linalg.norm(sol_enc.reshape(-1,1) - u_approx[i].reshape(-1,1))**2
        latent_error_norm_ = np.linalg.norm(sol_enc)**2
        latent_error += latent_error_
        latent_error_norm += latent_error_norm_

        print("iteration, latent error", i, np.sqrt(latent_error) / np.sqrt(latent_error_norm))
        print()

        # if i == nt: 
        #     space = NumpyVectorSpace(model.dims[0]*model.dims[1]*model.dims[2])
        #     fom.visualize(space.from_numpy(u_approx_full[i] - u_test.to_numpy()[:, i].reshape(-1,1)))

    relative_error = np.sqrt(error_sum) / np.sqrt(norm_sum)
    print(f'Relative error total: {relative_error}')

    relative_error_q = np.sqrt(error_sum_q) / np.sqrt(norm_sum_q)
    print(f'Relative error q: {relative_error_q}')

    relative_error_p = np.sqrt(error_sum_p) / np.sqrt(norm_sum_p)
    print(f'Relative error p: {relative_error_p}')

    relative_error_latent = np.sqrt(latent_error) / np.sqrt(latent_error_norm)
    print(f'Relative error latent: {relative_error_latent}')

    relative_error_file = script_dir / "test_relative_errors_wave.txt"
    with open(relative_error_file, "a") as f:
        f.write(f'Deep-Galerkin\t{p_red}\t{mu_test_val}\t{relative_error}\n')

if __name__ == '__main__':
    test_wave_2D()
