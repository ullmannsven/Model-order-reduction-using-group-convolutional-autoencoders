#!/usr/bin/env python

import time
import pickle
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from escnn import gspaces

from equiv_networks import  RotationUpsamplingGCNNAutoencoder2D, UpsamplingCNNAutoencoder2D, TrivialUpsamplingGCNNAutoencoder2D, RotationUpsamplingGCNN2D_TorchOnly
from equiv_networks.models.instationary.nonlinear_manifolds import NonlinearManifoldsMOR2D
from equiv_networks.early_stopping import SimpleEarlyStoppingScheduler
from scaling.scale import Scaler
from experiment_setup import WaveExperimentConfig

def train_wave_2D():
    config = WaveExperimentConfig(nt=500, visualize_q=True)
    p_red = 12

    Nx = config.Nx
    Ny = config.Ny
    dims = (2, Nx, Ny)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "snapshots_grid")
    os.makedirs(base_dir, exist_ok=True)

    arrays = []
    for mu in [0.5, 0.75, 1.0]:
        filename = os.path.join(base_dir, f'snapshots_{Nx}x{Ny}_mu{mu}')
        with open(filename, 'rb') as f:
            arr = pickle.load(f)['snapshots']
        arrays.append(arr)

    data_mat = np.vstack(arrays)
    
 
    print('Raw concatenated shape (flat):', data_mat.shape)

    T_total, n_space2 = data_mat.shape
    n_space = n_space2 // 2
    Nx = Ny = int(np.sqrt(n_space)) #TODO only works if Nx and Ny are the same

    # Slice q and p blocks
    q_flat = data_mat[:, :n_space]         
    p_flat = data_mat[:, n_space:]

    # Reshape each time slice to images and stack as (T, 2, Nx, Ny)
    snapshots_np = np.empty((T_total, 2, Nx, Ny), dtype=np.float64)
    for t in range(T_total):
        q_img = q_flat[t, :].reshape(Nx, Ny)
        p_img = p_flat[t, :].reshape(Nx, Ny)
        snapshots_np[t, 0, :, :] = q_img
        snapshots_np[t, 1, :, :] = p_img

    print('Reshaped snapshots shape (T, C, H, W):', snapshots_np.shape)

    q_min = snapshots_np[:, 0, :, :].min()
    q_max = snapshots_np[:, 0, :, :].max()
    p_min = snapshots_np[:, 1, :, :].min()
    p_max = snapshots_np[:, 1, :, :].max()

    scaling_dir = os.path.join(script_dir, "scaling")
    scaling_filename = os.path.join(scaling_dir, f"scaling_grid_{Nx}x{Ny}")
    with open(scaling_filename, 'wb') as f:
        pickle.dump({
            'min': {'q': float(q_min), 'p': float(p_min)},
            'max': {'q': float(q_max), 'p': float(p_max)},
            'shape': {'Nx': Nx, 'Ny': Ny}
        }, f)

    scaler = Scaler(dims=dims)
    snapshots_scaled = scaler.scale(torch.as_tensor(snapshots_np, dtype=torch.double, device="cpu"))
   
    # Build list of dicts
    snapshots = [{'u_full_step_shifted': snapshots_scaled[t, :, :, :]} for t in range(T_total)]

    # Shuffle and split
    np.random.shuffle(snapshots)
    train_cut = int(0.8 * len(snapshots))
    training_data = snapshots[:train_cut]
    validation_data = snapshots[train_cut:]

    print(f'Train/Val sizes: {len(training_data)} / {len(validation_data)}')
    print('One sample shape:', training_data[0]['u_full_step_shifted'].shape)
    
    checkpoints_dir = os.path.join(script_dir, "checkpoints")
    checkpoints_dir = Path(checkpoints_dir)
    os.makedirs(checkpoints_dir, exist_ok=True)

    current_time = time.strftime("%d_%m_%Y-%H_%M_%S")
    FILENAME = f'wave_2D_RotationUpsamplingGCNN_p_{p_red}_{Nx}x{Ny}_t_{current_time}'
    checkpoints_file = checkpoints_dir / f"{FILENAME}.pt"

    # parameters for the AE training
    number_of_epochs = 1000
    learning_rate = 0.0005
    batch_size = 20
    parameters_es_scheduler = {'checkpoint_filepath': checkpoints_file, 'patience': 100, 'delta': 1e-8, 'maximum_loss': None}

    network_parameters = {'encoder_channels': [2, 4, 8, 16, 32],
                        'decoder_channels': [32, 16, 8, 4, 2],
                        'encoder_fully_connected_layers_sizes': [4*p_red, p_red], 
                        'decoder_fully_connected_layers_sizes': [p_red, 4*p_red],
                        'encoder_kernel_sizes': 3,
                        'encoder_strides': 2, 
                        'encoder_paddings':1, 
                        'decoder_paddings':1,
                        'decoder_strides': 2,
                        'decoder_kernel_sizes': 3}

    payload = {'p_red': p_red, 'network_parameters': network_parameters}
    network_parameters_dir = os.path.join(script_dir, "network_parameters")
    network_parameters_dir = Path(network_parameters_dir)
    os.makedirs(network_parameters_dir, exist_ok=True)

    network_parameters_file = network_parameters_dir / f"{FILENAME}.pkl"

    with network_parameters_file.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    network_parameters['gspace'] = gspaces.rot2dOnR2(N=8) 
 
    model = NonlinearManifoldsMOR2D(network=RotationUpsamplingGCNNAutoencoder2D,
                                    scaler=scaler,
				    dims = dims,
                                    network_parameters=network_parameters,
                                    parameters_trainer={'optimizer': optim.Adam,
                                                        'learning_rate': learning_rate,    
                                                        'use_validation': True,
                                                        'es_scheduler': SimpleEarlyStoppingScheduler,
                                                        'parameters_es_scheduler': parameters_es_scheduler,
                                                        'loss_mode': None,
                                                        'loss_symplectic_fraction': 0.9,
                                                        'targets_are_normalized': True},
                                    loss_function = nn.MSELoss())

    validation_lass, training_loss = model.train(parameters_training={'training_data': training_data,
                                                                      'validation_data': validation_data,
                                                                      'number_of_epochs': number_of_epochs,
                                                                      'batch_size': batch_size,
                                                                      'nn_save_filepath': checkpoints_file})
    
    #validation_loss and training_loss could be used if one now runs this training procedure for different network architectures

if __name__ == '__main__':
    import os, torch, torch.multiprocessing as mp
    try: mp.set_sharing_strategy("file_system")
    except RuntimeError: pass
    try: mp.set_start_method("spawn", force=True)   # safer on clusters
    except RuntimeError: pass
    os.environ.setdefault("PYTORCH_SHM_TEMPDIR", os.path.expanduser("~/buildtmp"))
    os.makedirs(os.environ["PYTORCH_SHM_TEMPDIR"], exist_ok=True)
    train_wave_2D()
