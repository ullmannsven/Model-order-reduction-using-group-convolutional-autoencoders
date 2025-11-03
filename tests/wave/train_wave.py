#!/usr/bin/env python

import time
import pickle
import os
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from torchpdes.neuralnetworks.autoencoders import GCNNAutoencoder2D, RotationGCNNAutoencoder2D, CNNAutoencoder2D, RotationUpsamplingGCNNAutoencoder2D
from torchpdes.models.instationary.nonlinear_manifolds import NonlinearManifoldsMOR2D
from torchpdes.utilities.torch.early_stopping import SimpleEarlyStoppingScheduler
from torchpdes.pdes.examples.instationary import wave_2D
from scale import Scaler


def train_wave_2D(p_red):

    Nx = 51
    Ny = 51
    T = 1
    sig_pre = 2
    number_of_snapshots = 11

    fom = wave_2D(T=T, Nx=Nx, Ny=Ny, sig_pre=sig_pre, x_flow=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "snapshots_grid")
    os.makedirs(base_dir, exist_ok=True)

    arrays = []
    for i in range(number_of_snapshots):
        filename = os.path.join(base_dir, f'snapshots_{Nx}x{Ny}_sigpre_{sig_pre}_{number_of_snapshots}_{i}')
        with open(filename, 'rb') as f:
            arr = pickle.load(f)['snapshots']
        arrays.append(arr)

    data_mat = np.vstack(arrays)
    print('Raw concatenated shape (flat):', data_mat.shape)

    T_total, n_space2 = data_mat.shape
    n_space = n_space2 // 2
    
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
    dims = (2, Nx, Ny)

    eps = 1e-12
    q_min = snapshots_np[:, 0, :, :].min()
    q_max = snapshots_np[:, 0, :, :].max()
    p_min = snapshots_np[:, 1, :, :].min()
    p_max = snapshots_np[:, 1, :, :].max()

    scaling_filename = os.path.join(script_dir, f"scaling_grid_{Nx}x{Ny}")
    with open(scaling_filename, 'wb') as f:
        pickle.dump({
            'min': {'q': float(q_min), 'p': float(p_min)},
            'max': {'q': float(q_max), 'p': float(p_max)},
            'shape': {'Nx': Nx, 'Ny': Ny}
        }, f)

    scaler = Scaler(dims=dims)

    # Apply scaling: (x - min) / (max - min) for both q and p values
    # TODO i should use the scale method here?
    snapshots_scaled = snapshots_np.copy()
    #snapshots_scaled[:, 0, :, :] = (snapshots_np[:, 0, :, :] - q_min) / (q_max - q_min + eps)
    #snapshots_scaled[:, 1, :, :] = (snapshots_np[:, 1, :, :] - p_min) / (p_max - p_min + eps)

    # Build list of dicts
    snapshots = [{'u_full_step_shifted': torch.DoubleTensor(snapshots_scaled[t, :, :, :])} for t in range(T_total)]

    # Shuffle and split
    np.random.shuffle(snapshots)
    train_cut = int(0.9 * len(snapshots))
    training_data = snapshots[:train_cut]
    validation_data = snapshots[train_cut:]

    print(f'Train/Val sizes: {len(training_data)} / {len(validation_data)}')
    print('One sample shape:', training_data[0]['u_full_step_shifted'].shape)

    # folder to store checkpoint results and summaries of the runs
    run_dir = os.path.join(script_dir, "run")
    run_dir = Path(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    
    checkpoints_dir = os.path.join(script_dir, "checkpoints")
    checkpoints_dir = Path(checkpoints_dir)
    os.makedirs(checkpoints_dir, exist_ok=True)

    current_time = time.strftime("%d_%m_%Y-%H_%M_%S")
    FILENAME = f'wave_2D_CNN_p_{p_red}_t_{current_time}'
    log_file = run_dir / FILENAME
    checkpoints_file = checkpoints_dir / f"{FILENAME}.pt"

    # parameters for the AE training
    number_of_epochs = 1000
    learning_rate = 0.001
    batch_size = 80
    parameters_es_scheduler = {'checkpoint_filepath': checkpoints_file, 'patience': 100, 'delta': 1e-8, 'maximum_loss': None}

    network_parameters = {'encoder_channels': [2, 4, 8, 16, 32],
                        'decoder_channels': [32, 16, 8, 4, 2],
                        'encoder_fully_connected_layers_sizes': [64, p_red], 
                        'decoder_fully_connected_layers_sizes': [p_red, 64],
                        'encoder_strides': [2, 2, 2, 2, 1], 
                        'decoder_strides': [2, 2, 3, 2, 2], 
                        'decoder_paddings':[0, 0, 0, 2, 3], 
                        'decoder_kernel_sizes': [5, 5, 5, 5, 4]}
    
    payload = {'p_red': p_red, 'network_parameters': network_parameters}
    network_parameters_dir = os.path.join(script_dir, "network_parameters")
    network_parameters_dir = Path(network_parameters_dir)
    os.makedirs(network_parameters_dir, exist_ok=True)

    network_parameters_file = network_parameters_dir / f"{FILENAME}.pkl"

    with network_parameters_file.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
 
    model = NonlinearManifoldsMOR2D(full_order_model=fom,
                                    network=RotationUpsamplingGCNNAutoencoder2D,
                                    scaler=scaler,
                                    dims=dims,
                                    network_parameters=network_parameters,
                                    parameters_trainer={'optimizer': optim.Adam,
                                                        'learning_rate': learning_rate,
                                                        #TODO on server i use different lr-scheduler
                                                        'lr_scheduler': torch.optim.lr_scheduler.CosineAnnealingLR,
                                                        'parameters_lr_scheduler':{'T_max':number_of_epochs,'eta_min':1e-5},
                                                        'use_validation': True,
                                                        'es_scheduler': SimpleEarlyStoppingScheduler,
                                                        'parameters_es_scheduler': parameters_es_scheduler, 
                                                        'loss_mode': "weigths",
                                                        'loss_symplectic_fraction': 0.9,
                                                        'targets_are_normalized': True})

    validation_lass, training_loss = model.train(parameters_training={'training_data': training_data,
                                                                      'validation_data': validation_data,
                                                                      'number_of_epochs': number_of_epochs,
                                                                      'batch_size': batch_size,
                                                                      'log_filename': log_file,
                                                                      'nn_save_filepath': checkpoints_file})
    
    print("I AM DONE WITH TRAINING")
    
    #validation_loss and training_loss could be used if one now runs this training procedure for different network architectures

if __name__ == '__main__':
    train_wave_2D(p_red=30)
