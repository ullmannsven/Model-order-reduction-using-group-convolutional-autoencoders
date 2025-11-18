#!/usr/bin/env python

import numpy as np
import pickle
import os
from pathlib import Path

from pymor.basic import *

import torch

from torchpdes.neuralnetworks.autoencoders import GCNNAutoencoder2D, RotationGCNNAutoencoder2D, CNNAutoencoder2D, RotationUpsamplingGCNNAutoencoder2D
from torchpdes.models.instationary.nonlinear_manifolds import NonlinearManifoldsMOR2D
from scaling.scale import Scaler
from experiment_setup import WaveExperimentConfig, WaveExperiment

def test_wave_2D():

    # Configure experiment
    config = WaveExperimentConfig(x_flow=True, visualize_q=True)
    experiment = WaveExperiment(config)
    scaled_data = True
    p_red = 12

    timestep_factor = config.timestep_factor
    Nx = config.Nx
    Ny = config.Ny
  
    script_dir = os.path.dirname(os.path.abspath(__file__))
    nn_save_filepath = os.path.join(script_dir, "checkpoints")
    script_dir = Path(script_dir)
    filepaths = experiment.get_filepath_patterns(script_dir)
    nn_save_filepath = Path(nn_save_filepath) / "wave_2D_CNN_p_12_256x256_t_16_11_2025-13_56_01_scaled_larger_stride_nolr_00005.pt"

    network_parameters_dir = os.path.join(script_dir, "network_parameters")
    network_parameters_file = Path(network_parameters_dir) / "wave_2D_CNN_p_12_256x256_t_16_11_2025-13_56_01_scaled_larger_stride_nolr_00005.pkl"

    with Path(network_parameters_file).open("rb") as f:
        parameters = pickle.load(f)

    assert f"_{Nx}x{Ny}_" in str(nn_save_filepath)
    assert f"p_{p_red}_" in str(nn_save_filepath)

    scaler = Scaler(dims=config.dims)
    
    #RotationUpsamplingGCNNAutoencoder2D
    model = NonlinearManifoldsMOR2D(network=CNNAutoencoder2D, scaler=scaler, dims=config.dims, network_parameters=parameters['network_parameters'])
    model.load_neural_network(path=nn_save_filepath)
    model.network.eval()

    trainable = sum(p.numel() for p in model.network.parameters() if p.requires_grad)
    print("so viele parameter hat mein netz", trainable)

    mu_val = 1
    mu_test = experiment.fom.parameters.parse({'mu': mu_val})
    filename = filepaths['snapshots'] / "snapshots_256x256_sigpre_050_3_1_nt_1000_every_5_ts"
    with open(filename, 'rb') as f:
        arr = pickle.load(f)['snapshots']

    u_test = np.vstack(arr).T

    #NOTE: alternativ: compute u_test with x_flow False fom, but takes a lot longer
    if not config.x_flow:
        u_test = u_test.reshape(2, Nx, Ny, -1)
        u_test = np.rot90(u_test, k=-1, axes=(1,2)) #rotate countercockwise
        u_test = u_test.reshape(2*Nx*Ny, -1)


    # Compute initial condition and reference offset
    initial_state = experiment.get_initial_state(mu_val=mu_val)
    u_ref, _ = experiment.compute_reference_offset(model, mu_val=mu_val, scaled_data=scaled_data)
    u_ref = u_ref.reshape(-1,1)

    # some approximation results:
    amount_of_steps = int(config.T*config.nt/timestep_factor)
    errors = np.zeros((amount_of_steps, 1))
    errors_den = np.zeros((amount_of_steps, 1))

    errors_q = np.zeros((amount_of_steps, 1))
    errors_q_den = np.zeros((amount_of_steps, 1))
    errors_p = np.zeros((amount_of_steps, 1))
    errors_p_den = np.zeros((amount_of_steps, 1))

    for i in range(amount_of_steps):
        #Note dont subtract zero as we are working with loaded data where the zero has already been substracted
        sol_rot = u_test[:, i]

        if scaled_data:
            sol_rot_scaled = torch.as_tensor(scaler.scale(scaler.restrict(sol_rot)), dtype=torch.double, device="cpu").unsqueeze(0)
            sol_rot_enc = model.network.encode(sol_rot_scaled).detach().cpu().numpy()
            sol_rot_dec = model.network.decode(torch.as_tensor(sol_rot_enc, dtype=torch.double, device="cpu"))[0].detach().cpu().numpy()
            sol_rot_dec = scaler.prolongate(scaler.unscale(sol_rot_dec))

        else:
            sol_rot_enc = model.network.encode(torch.as_tensor(scaler.restrict(sol_rot), dtype=torch.double, device="cpu").unsqueeze(0)).detach().cpu().numpy()
            sol_rot_dec = model.network.decode(torch.as_tensor(sol_rot_enc, dtype=torch.double, device="cpu"))[0].detach().cpu().numpy()
            sol_rot_dec = scaler.prolongate(sol_rot_dec)

        if i == 50: 
            space = NumpyVectorSpace(model.dims[0]*model.dims[1]*model.dims[2])
            experiment.fom.visualize(space.from_numpy(sol_rot_dec.reshape(-1,1) + u_ref))
            experiment.fom.visualize(space.from_numpy(u_test[:, i].reshape(-1,1) + initial_state))
            experiment.fom.visualize(space.from_numpy(u_test[:, i].reshape(-1,1) + initial_state) - (space.from_numpy(sol_rot_dec.reshape(-1,1) + u_ref)))
        
        errors[i, 0] = np.linalg.norm(sol_rot.reshape(-1,1) - (sol_rot_dec.reshape(-1,1)))**2
        errors_den[i, 0] = np.linalg.norm(u_test[:, i])**2

        errors_q[i, 0] = np.linalg.norm(sol_rot.reshape(-1,1)[:Nx*Ny, :] - (sol_rot_dec.reshape(-1,1)[:Nx*Ny, :]))**2
        errors_q_den[i, 0] = np.linalg.norm(u_test[:Nx*Ny, i])**2

        errors_p[i, 0] = np.linalg.norm(sol_rot.reshape(-1,1)[Nx*Ny:, :] - (sol_rot_dec.reshape(-1,1)[Nx*Ny:, :]))**2
        errors_p_den[i, 0] = np.linalg.norm(u_test[Nx*Ny:, i])**2

    print("error", np.sqrt(np.sum(errors, axis=0) / np.sum(errors_den, axis=0)))
    print("error q", np.sqrt(np.sum(errors_q, axis=0) / np.sum(errors_q_den, axis=0)))
    print("error p", np.sqrt(np.sum(errors_p, axis=0) / np.sum(errors_p_den, axis=0)))


if __name__ == '__main__':
    test_wave_2D()    

   