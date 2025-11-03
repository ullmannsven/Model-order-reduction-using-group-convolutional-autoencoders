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

from torchpdes.neuralnetworks.autoencoders import GCNNAutoencoder2D, RotationGCNNAutoencoder2D, CNNAutoencoder2D, RotationUpsamplingGCNNAutoencoder2D
from torchpdes.models.instationary.nonlinear_manifolds import NonlinearManifoldsMOR2D
from torchpdes.pdes.instationary import wave_2D
from scale import Scaler

def test_wave_2D():

    p_red = 16
    Nx = 256
    Ny = 256
    sig_pre = 0.5
    timestep_factor = 5
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    nn_save_filepath = os.path.join(script_dir, "checkpoints")
    script_dir = Path(script_dir)
    nn_save_filepath = Path(nn_save_filepath) / "wave_2D_CNN_256x256_p_16_t_01_11_2025-14_02_55_scaled_circular_padding_warmrestarts.pt"

    network_parameters_dir = os.path.join(script_dir, "network_parameters")
    network_parameters_file = Path(network_parameters_dir) / "wave_2D_CNN_256x256_p_16_t_01_11_2025-14_02_55_scaled_circular_padding_warmrestarts.pkl"

    with Path(network_parameters_file).open("rb") as f:
        parameters = pickle.load(f)

    assert f"_{Nx}x{Ny}_" in str(nn_save_filepath)
    assert f"p_{p_red}_" in str(nn_save_filepath)

    x_flow = True
    T = 1
    nt = 1000 #number of timesteps per second
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

    mu_test_val = 1.25
    mu_test = fom.parameters.parse({'mu': mu_test_val})
    print(f'Solving for test parameter = {mu_test} ... ')
    u_test = fom.solve(mu_test)
    print("done with FOM solve")

    # some approximation results:
    amount_of_steps = int(T*nt/timestep_factor)
    errors = np.zeros((amount_of_steps, 1))
    errors_den = np.zeros((amount_of_steps, 1))

    errors_q =  np.zeros((amount_of_steps, 1))
    errors_q_den =  np.zeros((amount_of_steps, 1))
    errors_p =  np.zeros((amount_of_steps, 1))
    errors_p_den =  np.zeros((amount_of_steps, 1))

    for i in range(amount_of_steps):
        sol_rot = u_test.to_numpy()[:, timestep_factor*i] - u_test.to_numpy()[:, 0]

        print("iteration", i)
        
        # sol_rot_enc = model.network.encode(torch.as_tensor(scaler.restrict(sol_rot), dtype=torch.double, device="cpu").unsqueeze(0))
        # sol_rot_dec = model.network.decode(sol_rot_enc).detach().cpu().numpy()
        # sol_rot_dec = scaler.prolongate(sol_rot_dec)
        # if i == 50: 
        #     space = NumpyVectorSpace(model.dims[0]*model.dims[1]*model.dims[2])
        #     fom.visualize(space.from_numpy(sol_rot_dec.reshape(-1,1) + u_test.to_numpy()[:, 0].reshape(-1,1) - u_test.to_numpy()[:, timestep_factor*i].reshape(-1,1)))

        sol_rot_scaled = torch.as_tensor(scaler.scale(scaler.restrict(sol_rot)), dtype=torch.double, device="cpu").unsqueeze(0)
        sol_rot_enc = model.network.encode(sol_rot_scaled).detach().cpu().numpy()
        sol_rot_dec = model.network.decode(torch.as_tensor(sol_rot_enc, dtype=torch.double, device="cpu"))[0].detach().cpu().numpy()
        sol_rot_dec = scaler.prolongate(scaler.unscale(sol_rot_dec))
        
        errors[i, 0] = np.linalg.norm(sol_rot.reshape(-1,1) - sol_rot_dec.reshape(-1,1))**2
        errors_den[i, 0] = np.linalg.norm(u_test.to_numpy()[:, timestep_factor*i])**2

        errors_q[i, 0] = np.linalg.norm(sol_rot.reshape(-1,1)[:Nx*Ny] - sol_rot_dec.reshape(-1,1)[:Nx*Ny])**2
        errors_q_den[i, 0] = np.linalg.norm(u_test.to_numpy()[:Nx*Ny, timestep_factor*i])**2

        errors_p[i, 0] = np.linalg.norm(sol_rot.reshape(-1,1)[Nx*Nx:] - sol_rot_dec.reshape(-1,1)[Nx*Ny:])**2
        errors_p_den[i, 0] = np.linalg.norm(u_test.to_numpy()[Nx*Ny:, timestep_factor*i])**2

    print("error", np.sqrt(np.sum(errors, axis=0) / np.sum(errors_den, axis=0)))
    print("error q", np.sqrt(np.sum(errors_q, axis=0) / np.sum(errors_q_den, axis=0)))
    print("error p", np.sqrt(np.sum(errors_p, axis=0) / np.sum(errors_p_den, axis=0)))


if __name__ == '__main__':
    test_wave_2D()    

   