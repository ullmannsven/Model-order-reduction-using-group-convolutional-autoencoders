#!/usr/bin/env python

import numpy as np
import pickle
import os
from pathlib import Path
import csv

from pymor.basic import *

import torch
from escnn import gspaces

from equiv_networks.autoencoders import RotationUpsamplingGCNNAutoencoder2D, UpsamplingCNNAutoencoder2D
from equiv_networks.equiv_ae import RotationUpsamplingGCNN2D_TorchOnly
from equiv_networks.models.instationary.nonlinear_manifolds import NonlinearManifoldsMOR2D
from scaling.scale import Scaler
from experiment_setup import WaveExperimentConfig, WaveExperiment

def proj_error_AE():

    # Configure experiment
    config = WaveExperimentConfig(x_flow=False, visualize_q=True, nt=500, timestep_factor=1)
    experiment = WaveExperiment(config)
    scaled_data = True

    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir = Path(script_dir)
    filepaths = experiment.get_filepath_patterns(script_dir)
    
    proj_errors = []

    for p_red in [12]:
       
        timestep_factor = config.timestep_factor
        Nx = config.Nx
        Ny = config.Ny
    
        nn_save_filepath = os.path.join(script_dir, "checkpoints")
        nn_save_filepath = Path(nn_save_filepath) / f"wave_2D_RotationUpsamplingGCNN_C8_p_12_256x256_t_09_02_2026-11_14_30.pt"

        network_parameters_dir = os.path.join(script_dir, "network_parameters")
        network_parameters_file = Path(network_parameters_dir) / f"wave_2D_RotationUpsamplingGCNN_C8_p_12_256x256_t_09_02_2026-11_14_30.pkl"
    
        with Path(network_parameters_file).open("rb") as f:
            parameters = pickle.load(f)

        assert f"_{Nx}x{Ny}" in str(nn_save_filepath)
        assert f"p_{p_red}_" in str(nn_save_filepath)

        scaler = Scaler(dims=config.dims)
        parameters['network_parameters']['gspace'] = gspaces.rot2dOnR2(N=8)

        model = NonlinearManifoldsMOR2D(network=RotationUpsamplingGCNNAutoencoder2D, scaler=scaler, dims=config.dims, network_parameters=parameters['network_parameters'])
        model.load_neural_network(path=nn_save_filepath)
        model.network.eval()

        #trainable = sum(p.numel() for p in model.network.parameters() if p.requires_grad)
        #print("so viele parameter hat mein netz", trainable)

        #print("parameter", parameters['network_parameters'])

        mu_val = 0.6
        filename = filepaths['snapshots'] / "snapshots_256x256_060_nt_500"
        with open(filename, 'rb') as f:
            arr = pickle.load(f)['snapshots']
        u_test = np.vstack(arr).T

        #NOTE: alternativ: compute u_test with x_flow False fom, but takes a lot longer and is not required for 90 degree rotation
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

            if i == 300: 
               space = NumpyVectorSpace(model.dims[0]*model.dims[1]*model.dims[2])
               experiment.fom.visualize(space.from_numpy(sol_rot_dec.reshape(-1,1) + u_ref))
               experiment.fom.visualize(space.from_numpy(u_test[:, i].reshape(-1,1) + initial_state))
               experiment.fom.visualize(space.from_numpy(u_test[:, i].reshape(-1,1) + initial_state) - space.from_numpy(sol_rot_dec.reshape(-1,1) + u_ref))
            
            errors[i, 0] = np.linalg.norm(sol_rot.reshape(-1,1) - (sol_rot_dec.reshape(-1,1)))**2
            errors_den[i, 0] = np.linalg.norm(u_test[:, i])**2

            errors_q[i, 0] = np.linalg.norm(sol_rot.reshape(-1,1)[:Nx*Ny, :] + initial_state[:Nx*Ny, :] - (sol_rot_dec.reshape(-1,1)[:Nx*Ny, :] + u_ref[:Nx*Ny, :]))**2
            errors_q_den[i, 0] = np.linalg.norm(u_test[:Nx*Ny, i] + initial_state[:Nx*Ny, 0])**2

            errors_p[i, 0] = np.linalg.norm(sol_rot.reshape(-1,1)[Nx*Ny:, :] + initial_state[Nx*Ny:, :] - (sol_rot_dec.reshape(-1,1)[Nx*Ny:, :] + u_ref[Nx*Ny:, :]))**2
            errors_p_den[i, 0] = np.linalg.norm(u_test[Nx*Ny:, i]+ initial_state[Ny*Ny:, 0])**2

        print("error", np.sqrt(np.sum(errors, axis=0) / np.sum(errors_den, axis=0)))
        print("error q", np.sqrt(np.sum(errors_q, axis=0) / np.sum(errors_q_den, axis=0)))
        print("error p", np.sqrt(np.sum(errors_p, axis=0) / np.sum(errors_p_den, axis=0)))

        proj_errors.append((p_red, np.sqrt(np.sum(errors, axis=0) / np.sum(errors_den, axis=0))[0]))

    print(proj_errors)

    # Target folder (relative or absolute)
    # out_file = filepaths['AE_results'] / f"proj_error_ae_GCNN_mu{mu_val}.csv"

    # with open(out_file, "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["x", "y"])
    #     writer.writerows(proj_errors)

if __name__ == '__main__':
    proj_error_AE()    

   