#!/usr/bin/env python
"""
Train an Autoencoder on the 2D wave equation experiment.

Usage:
    python train_wave_2D.py --ae_name AE_NAME [--p_red P] [--xflow] [--both_directions]
                            [--epochs N] [--lr LR] [--batch_size B]

Arguments:
    --ae_name           Name of the autoencoder architecture.
                        Choices:
                            RotationUpsamplingGCNN_C4   -> RotationUpsamplingGCNNAutoencoder2D (N=4)
                            RotationUpsamplingGCNN_C8   -> RotationUpsamplingGCNNAutoencoder2D (N=8)
                            UpsamplingCNN               -> UpsamplingCNNAutoencoder2D
                            TrivialUpsamplingGCNN       -> TrivialUpsamplingGCNNAutoencoder2D
    --p_red             Reduced dimension (default: 12)
    --xflow             Use x-direction flow (default: True)
    --both_directions   Augment training data with rotated snapshots (default: False)
    --epochs            Number of training epochs (default: 1000)
    --lr                Learning rate (default: 0.0005)
    --batch_size        Batch size (default: 20)

Examples:
    # C4 equivariant network, default settings
    python train_wave_2D.py --ae_name RotationUpsamplingGCNN_C4

    # CNN baseline, p_red=8, both directions
    python train_wave_2D.py --ae_name UpsamplingCNN --p_red 8 --both_directions

    # C8 network, custom learning rate and epochs
    python train_wave_2D.py --ae_name RotationUpsamplingGCNN_C8 --p_red 16 --lr 0.001 --epochs 500

    # y-flow
    python train_wave_2D.py --ae_name UpsamplingCNN --no-xflow
"""

import argparse
import time
import pickle
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from escnn import gspaces

from equiv_networks.autoencoders import (
    RotationUpsamplingGCNNAutoencoder2D,
    UpsamplingCNNAutoencoder2D
)
from equiv_networks.models.nonlinear_manifolds import NonlinearManifoldsMOR2D
from equiv_networks.early_stopping import SimpleEarlyStoppingScheduler
from scaling.scale import Scaler
from experiment_setup import WaveExperimentConfig

AE_REGISTRY = {
    'RotationUpsamplingGCNN': {
        'class': RotationUpsamplingGCNNAutoencoder2D,
        'gspace': lambda: gspaces.rot2dOnR2(N=4),
    },
    'RotationUpsamplingGCNN_C8': {
        'class': RotationUpsamplingGCNNAutoencoder2D,
        'gspace': lambda: gspaces.rot2dOnR2(N=8),
    },
    'UpsamplingCNN': {
        'class': UpsamplingCNNAutoencoder2D,
        'gspace': None,
    },
}


def train_wave_2D(ae_name, p_red=12, x_flow=True, both_directions=False,
                  number_of_epochs=1000, learning_rate=0.0005, batch_size=20):

    config = WaveExperimentConfig(x_flow=x_flow, nt=500, visualize_q=True)

    Nx = config.Nx
    Ny = config.Ny
    dims = (2, Nx, Ny)

    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    base_dir = script_dir / "snapshots_grid"
    os.makedirs(base_dir, exist_ok=True)

    arrays = []
    for mu in [0.5, 0.75, 1]:
        filename = base_dir / f'snapshots_{Nx}x{Ny}_{mu}_nt_{config.nt}'
        with open(filename, 'rb') as f:
            arr = pickle.load(f)['snapshots']
        arrays.append(arr)

    data_mat = np.vstack(arrays)

    if both_directions:
        data_mat_rot = data_mat.T.reshape(2, Nx, Ny, -1)
        data_mat_rot = np.rot90(data_mat_rot, k=-1, axes=(1, 2))
        data_mat_rot = data_mat_rot.reshape(2 * Nx * Ny, -1)
        data_mat = np.concatenate((data_mat, data_mat_rot.T), axis=0)

    print('Raw concatenated shape (flat):', data_mat.shape)

    T_total, n_space2 = data_mat.shape
    n_space = n_space2 // 2
    Nx = Ny = int(np.sqrt(n_space))

    q_flat = data_mat[:, :n_space]
    p_flat = data_mat[:, n_space:]

    snapshots_np = np.empty((T_total, 2, Nx, Ny), dtype=np.float64)
    for t in range(T_total):
        snapshots_np[t, 0, :, :] = q_flat[t, :].reshape(Nx, Ny)
        snapshots_np[t, 1, :, :] = p_flat[t, :].reshape(Nx, Ny)

    print('Reshaped snapshots shape (T, C, H, W):', snapshots_np.shape)

    scaling_filename = script_dir / f"scaling_grid_{Nx}x{Ny}_new_idea"
    with open(scaling_filename, 'wb') as f:
        pickle.dump({
            'min': {'q': float(snapshots_np[:, 0].min()), 'p': float(snapshots_np[:, 1].min())},
            'max': {'q': float(snapshots_np[:, 0].max()), 'p': float(snapshots_np[:, 1].max())},
            'shape': {'Nx': Nx, 'Ny': Ny},
        }, f)

    scaler = Scaler(dims=dims)
    snapshots_scaled = scaler.scale(torch.as_tensor(snapshots_np, dtype=torch.double, device="cpu"))

    snapshots = [{'u_full_step_shifted': snapshots_scaled[t]} for t in range(T_total)]

    np.random.shuffle(snapshots)
    train_cut = int(0.8 * len(snapshots))
    training_data = snapshots[:train_cut]
    validation_data = snapshots[train_cut:]

    print(f'Train/Val sizes: {len(training_data)} / {len(validation_data)}')
    print('One sample shape:', training_data[0]['u_full_step_shifted'].shape)

    run_dir = script_dir / "run"
    checkpoints_dir = script_dir / "checkpoints"
    network_parameters_dir = script_dir / "network_parameters"
    for d in [run_dir, checkpoints_dir, network_parameters_dir]:
        os.makedirs(d, exist_ok=True)

    current_time = time.strftime("%d_%m_%Y-%H_%M_%S")
    stem = f"wave_2D_{ae_name}_p_{p_red}_{Nx}x{Ny}_t_{current_time}"
    log_file = run_dir / stem
    checkpoints_file = checkpoints_dir / f"{stem}.pt"

    network_parameters = {
        'encoder_channels': [2, 4, 8, 16, 32],
        'decoder_channels': [32, 16, 8, 4, 2],
        'encoder_fully_connected_layers_sizes': [4 * p_red, p_red],
        'decoder_fully_connected_layers_sizes': [p_red, 4 * p_red],
        'encoder_kernel_sizes': 3,
        'encoder_strides': 2,
        'encoder_paddings': 1,
        'decoder_paddings': 1,
        'decoder_strides': 2,
        'decoder_kernel_sizes': 3,
    }

    ae_entry = AE_REGISTRY[ae_name]
    if ae_entry['gspace'] is not None:
        network_parameters['gspace'] = ae_entry['gspace']()

    payload = {'p_red': p_red, 'network_parameters': network_parameters}
    with (network_parameters_dir / f"{stem}.pkl").open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    parameters_es_scheduler = {
        'checkpoint_filepath': checkpoints_file,
        'patience': 100,
        'delta': 1e-8,
        'maximum_loss': None,
    }

    model = NonlinearManifoldsMOR2D(
        network=ae_entry['class'],
        scaler=scaler,
        dims=dims,
        network_parameters=network_parameters,
        parameters_trainer={
            'optimizer': optim.Adam,
            'learning_rate': learning_rate,
            'use_validation': True,
            'es_scheduler': SimpleEarlyStoppingScheduler,
            'parameters_es_scheduler': parameters_es_scheduler,
            'loss_mode': None,
            'loss_symplectic_fraction': 0.9,
            'targets_are_normalized': True,
        },
        loss_function=nn.MSELoss(),
    )

    _, _ = model.train(parameters_training={
        'training_data': training_data,
        'validation_data': validation_data,
        'number_of_epochs': number_of_epochs,
        'batch_size': batch_size,
        'log_filename': log_file,
        'nn_save_filepath': checkpoints_file,
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train an Autoencoder on the 2D wave equation experiment.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument('--ae_name', type=str, required=True, choices=list(AE_REGISTRY.keys()), help='Autoencoder architecture name (determines network class and gspace)')
    parser.add_argument('--p_red', type=int, default=12, help='Reduced dimension (default: 12)')
    parser.add_argument('--xflow', action=argparse.BooleanOptionalAction, default=True, help='Use x-direction flow (default: True)')
    parser.add_argument('--both_directions', action='store_true', default=False, help='Augment training data with rotated snapshots (default: False)')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate (default: 0.0005)')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size (default: 20)')

    args = parser.parse_args()
    train_wave_2D(ae_name=args.ae_name, p_red=args.p_red, x_flow=args.xflow, both_directions=args.both_directions, number_of_epochs=args.epochs, learning_rate=args.lr, batch_size=args.batch_size)