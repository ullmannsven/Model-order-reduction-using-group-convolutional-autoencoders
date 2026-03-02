#!/usr/bin/env python
"""
Test 2D wave equation with Deep-Galerkin method.

Usage:
    python test_wave_deep_galerkin.py --ae_name AE_NAME [--mu_val MU]
                                      [--p_red N] [--scaled_data]
                                      [--symplectic] [--visualize]

Arguments:
    --ae_name       Name of the autoencoder architecture.
                    Choices:
                        RotationUpsamplingGCNN_C4   -> RotationUpsamplingGCNNAutoencoder2D (N=4)
                        RotationUpsamplingGCNN_C8   -> RotationUpsamplingGCNNAutoencoder2D (N=8)
                        UpsamplingCNN               -> UpsamplingCNNAutoencoder2D
                        TrivialUpsamplingGCNN       -> TrivialUpsamplingGCNNAutoencoder2D

    --mu_val        Test parameter value (default: 0.8)
    --p_red         Reduced dimension (default: 8)
    --scaled_data   Use scaled data (default: True). 
    --symplectic    Enable symplectic integration (default: False)
    --visualize     Enable visualization during timestepping.
    --save_data     Save reduction errors to a CSV file (default: False)

Checkpoint naming convention:
    wave_2D_{ae_name}_p_{p_red}_{Nx}x{Ny}.pt
    wave_2D_{ae_name}_p_{p_red}_{Nx}x{Ny}.pkl

Examples:
    # C8 network, default settings
    python test_wave_deep_galerkin.py --ae_name RotationUpsamplingGCNN_C8

    # CNN baseline, different mu
    python test_wave_deep_galerkin.py --ae_name UpsamplingCNN --mu_val 0.6 --p_red 12

    # With symplectic integration and visualization
    python test_wave_deep_galerkin.py --ae_name RotationUpsamplingGCNN_C4 --symplectic --visualize
"""

import argparse
import time
import numpy as np
import pickle
import os
from pathlib import Path
import csv
from pymor.basic import *

import torch
from escnn import gspaces

from equiv_networks.autoencoders import (
    RotationUpsamplingGCNNAutoencoder2D,
    UpsamplingCNNAutoencoder2D,
    TrivialUpsamplingGCNNAutoencoder2D,
)
from equiv_networks.models.nonlinear_manifolds import NonlinearManifoldsMOR2D
from equiv_networks.models.deep_galerkin_utilities_IMR import Galerkin_quasi_newton
from scaling.scale import Scaler
from experiment_setup import WaveExperiment, WaveExperimentConfig


AE_REGISTRY = {
    'RotationUpsamplingGCNN_C4': {
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
    'TrivialUpsamplingGCNN': {
        'class': TrivialUpsamplingGCNNAutoencoder2D,
        'gspace': None,
    },
}


def test_wave_deep_galerkin(ae_name, mu_val=0.8, p_red=8, scaled_data=True, symplectic=False, visualize=False, save_data=False):
    """Test 2D wave equation with Deep-Galerkin method."""

    if ae_name not in AE_REGISTRY:
        raise ValueError(f"Unknown ae_name '{ae_name}'. Choose from: {list(AE_REGISTRY.keys())}")
    ae_entry = AE_REGISTRY[ae_name]

    config = WaveExperimentConfig(x_flow=False, nt=500, timestep_factor=1, visualize_q=visualize)
    experiment = WaveExperiment(config)

    Nx = config.Nx
    Ny = config.Ny
    grid = f"{Nx}x{Ny}"

    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    filepaths = experiment.get_filepath_patterns(script_dir)

    stem = f"wave_2D_{ae_name}_p_{p_red}_{grid}"
    nn_save_filepath = filepaths['checkpoints'] / f"{stem}.pt"
    network_parameters_file = filepaths['network_parameters'] / f"{stem}.pkl"

    with open(network_parameters_file, "rb") as f:
        parameters = pickle.load(f)

    scaler = Scaler(dims=config.dims)
    if ae_entry['gspace'] is not None:
        parameters['network_parameters']['gspace'] = ae_entry['gspace']()

    model = NonlinearManifoldsMOR2D(
        network=ae_entry['class'],
        scaler=scaler,
        dims=config.dims,
        network_parameters=parameters['network_parameters'],
    )
    model.load_neural_network(path=nn_save_filepath)
    model.network.eval()

    mu_tag = f"{mu_val:.2f}".replace('.', '')
    mu_test = experiment.fom.parameters.parse({'mu': mu_val})
    filename = filepaths['snapshots'] / f"snapshots_{grid}_{mu_tag}_nt_{config.nt}"
    with open(filename, 'rb') as f:
        arr = pickle.load(f)['snapshots']
    u_test = np.vstack(arr).T

    if not config.x_flow:
        u_test = u_test.reshape(2, Nx, Ny, -1)
        u_test = np.rot90(u_test, k=-1, axes=(1, 2))
        u_test = u_test.reshape(2 * Nx * Ny, -1)

    initial_state = experiment.get_initial_state(mu_val=mu_val)
    u_ref, u_0_hat = experiment.compute_reference_offset(model, mu_val=mu_val, scaled_data=scaled_data)

    u_approx = [u_0_hat]
    u_approx_full = [initial_state]
    u_test = u_test + initial_state

    print("Starting timestepping for ROM...")
    for i in range(config.n_timesteps):
        tic = time.time()
        t = (i + 1) * config.dt
        print(f'Time: {t:.3f}')

        u_n1 = u_approx[-1]
        u_new = Galerkin_quasi_newton(model, u_n1, mu_test, config.dt, experiment.fom, u_ref, scaled_data, symplectic=symplectic, tol=1e-8)
        u_approx.append(u_new)

        decode_u_new = model.network.decode(torch.as_tensor(u_new, dtype=torch.double, device="cpu"))[0].detach().cpu().numpy()
        if scaled_data:
            decode_u_new = scaler.prolongate(scaler.unscale(decode_u_new))
        else:
            decode_u_new = scaler.prolongate(decode_u_new)

        u_approx_full.append(u_ref + decode_u_new)

        if visualize and i in (0, 50, 100):
            space2 = NumpyVectorSpace(Nx * Ny * 2)
            experiment.fom.visualize(space2.make_array(u_approx_full[i]))

        print(f"Step took {time.time() - tic:.2f}s")

    print("\n" + "=" * 60)
    print("Error Metrics:")
    print("=" * 60)

    metrics = experiment.compute_error_metrics(u_approx_full, u_test)

    print(f"Relative error (total):  {metrics['relative_error_total']:.6e}")
    print(f"Relative error (q):      {metrics['relative_error_q']:.6e}")
    print(f"Relative error (p):      {metrics['relative_error_p']:.6e}")

    
    if save_data:
        out_file = filepaths['results'] / f"reduction_error_{ae_name}_mu{mu_val}.csv"
        file_exists = out_file.exists()
        with open(out_file, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["x", "y"])
            writer.writerow([p_red, metrics['relative_error_total']])
        print(f"Saved CSV to: {out_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test 2D wave equation with Deep-Galerkin method.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--ae_name', type=str, required=True, choices=list(AE_REGISTRY.keys()), help='Autoencoder architecture name')
    parser.add_argument('--mu_val', type=float, default=0.8, help='Test parameter value mu (default: 0.8)')
    parser.add_argument('--p_red', type=int, default=8, help='Reduced dimension (default: 8)')
    parser.add_argument('--scaled_data', action=argparse.BooleanOptionalAction, default=True, help='Use scaled data (default: True)')
    parser.add_argument('--symplectic', action='store_true', default=False, help='Enable symplectic integration')
    parser.add_argument('--visualize', action='store_true', default=False, help='Enable visualization during timestepping')
    parser.add_argument('--save_data', action='store_true', default=False, help='Save data to CSV file')

    args = parser.parse_args()
    test_wave_deep_galerkin(ae_name=args.ae_name, mu_val=args.mu_val, p_red=args.p_red, scaled_data=args.scaled_data, symplectic=args.symplectic, visualize=args.visualize, save_data=args.save_data)