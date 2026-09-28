#!/usr/bin/env python
"""
Compute projection errors for a trained Autoencoder on the wave equation experiment.

Usage:
    python proj_error_AE_rotated.py --ae_name AE_NAME [--p_red P [P ...]] [--mu_val MU] [--scaled_data] [--rotated] [--visualize] [--write_csv]

Arguments:
    --ae_name       Name of the autoencoder architecture. Determines which network
                    class is used and how checkpoint files are located.
                    Choices:
                        RotationUpsamplingGCNN_C4   -> RotationUpsamplingGCNNAutoencoder2D (N=4)
                        RotationUpsamplingGCNN_C8   -> RotationUpsamplingGCNNAutoencoder2D (N=8)
                        UpsamplingCNN               -> UpsamplingCNNAutoencoder2D
    --p_red         One or more reduced dimensions to evaluate (default: 4 8 12 16)
    --mu_val        Test parameter value (default: 0.6)
    --scaled_data   Use scaled data (default: True)
    --rotated       Use rotated snapshot data (default: False)
    --visualize     Enable visualization during timestepping (default: False)
    --write_csv     Write projection errors to a CSV file (default: False)

Examples:
    # C4 equivariant network, default p_red values
    python proj_error_AE_rotated.py --ae_name RotationUpsamplingGCNN_C4

    # C8 equivariant network, single p_red, write CSV
    python proj_error_AE_rotated.py --ae_name RotationUpsamplingGCNN_C8 --p_red 12 --write_csv

    # CNN baseline, multiple p_red values, different mu, visualize
    python proj_error_AE_rotated.py --ae_name UpsamplingCNN --p_red 4 8 16 --mu_val 0.75 --visualize

"""

import argparse
import numpy as np
import pickle
import os
from pathlib import Path
import csv

from pymor.basic import *

import torch
from escnn import gspaces

from equiv_networks.autoencoders import RotationUpsamplingGCNNAutoencoder2D
from equiv_networks.models.nonlinear_manifolds import NonlinearManifoldsMOR2D
from scaling.scale import Scaler
from experiment_setup import WaveExperimentConfig, WaveExperiment

AE_REGISTRY = {
    'RotationUpsamplingGCNN': {
        'class': RotationUpsamplingGCNNAutoencoder2D,
        'gspace': lambda: gspaces.rot2dOnR2(N=4),
    },
    'RotationUpsamplingGCNN_C8': {
        'class': RotationUpsamplingGCNNAutoencoder2D,
        'gspace': lambda: gspaces.rot2dOnR2(N=8),
    },
}


def proj_error_AE(ae_name, p_red_values, mu_val=0.6, scaled_data=True, rotated=False, visualize=False, write_csv=False):

    config = WaveExperimentConfig(rotated=rotated, visualize_q=True, nt=500, timestep_factor=1)
    experiment = WaveExperiment(config)

    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    filepaths = experiment.get_filepath_patterns(script_dir)

    Nx = config.Nx
    Ny = config.Ny
    timestep_factor = config.timestep_factor
    grid = f"{Nx}x{Ny}"

    checkpoint_dir = script_dir / "checkpoints"

    ae_entry = AE_REGISTRY[ae_name]
    network_class = ae_entry['class']

    proj_errors = []

    for p_red in p_red_values:
        print(f"\n--- p_red = {p_red} ---")
        stem = f"wave_2D_{ae_name}_p_{p_red}_{grid}"
        nn_save_filepath = checkpoint_dir / f"{stem}.pt"
        network_parameters_file = script_dir / "network_parameters" / f"{stem}.pkl"

        with Path(network_parameters_file).open("rb") as f:
            parameters = pickle.load(f)

        scaler = Scaler(dims=config.dims)

        if ae_entry['gspace'] is not None:
            parameters['network_parameters']['gspace'] = ae_entry['gspace']()

        model = NonlinearManifoldsMOR2D(
            network=network_class,
            scaler=scaler,
            dims=config.dims,
            network_parameters=parameters['network_parameters'],
        )

        model.load_neural_network(path=nn_save_filepath)
        model.network.eval()

        mu_tag = f"{mu_val:.2f}".replace('.', '')
        rotated_tag = "_rotated" if rotated else ""
        filename = filepaths['snapshots'] / f"snapshots_45wave{rotated_tag}_{grid}_mu_{mu_tag}_nt_{config.nt}"
        with open(filename, 'rb') as f:
            arr = pickle.load(f)['snapshots']
        u_test = np.vstack(arr).T

        u_ref, initial_state, _ = experiment.compute_reference_offset(model, mu_val=mu_val, scaled_data=scaled_data)
        u_ref = u_ref.reshape(-1, 1)

        amount_of_steps = int(config.T * config.nt / timestep_factor)
        errors = np.zeros((amount_of_steps, 1))
        errors_den = np.zeros((amount_of_steps, 1))
        errors_q = np.zeros((amount_of_steps, 1))
        errors_q_den = np.zeros((amount_of_steps, 1))
        errors_p = np.zeros((amount_of_steps, 1))
        errors_p_den = np.zeros((amount_of_steps, 1))

        for i in range(amount_of_steps):
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

            if visualize and i == 100:
                space = NumpyVectorSpace(config.Nx * config.Ny * 2)
                experiment.fom.visualize(space.from_numpy(sol_rot_dec.reshape(-1, 1) + u_ref))
                experiment.fom.visualize(space.from_numpy(u_test[:, i].reshape(-1, 1) + initial_state))
                experiment.fom.visualize(space.from_numpy(u_test[:, i].reshape(-1, 1) + initial_state) - space.from_numpy(sol_rot_dec.reshape(-1, 1) + u_ref))

            errors[i, 0] = np.linalg.norm(sol_rot.reshape(-1, 1) - sol_rot_dec.reshape(-1, 1)) ** 2
            errors_den[i, 0] = np.linalg.norm(u_test[:, i]) ** 2

            errors_q[i, 0] = np.linalg.norm(sol_rot.reshape(-1, 1)[:Nx*Ny, :] + initial_state[:Nx*Ny, :] - (sol_rot_dec.reshape(-1, 1)[:Nx*Ny, :] + u_ref[:Nx*Ny, :])) ** 2
            errors_q_den[i, 0] = np.linalg.norm(u_test[:Nx*Ny, i] + initial_state[:Nx*Ny, 0]) ** 2

            errors_p[i, 0] = np.linalg.norm(sol_rot.reshape(-1, 1)[Nx*Ny:, :] + initial_state[Nx*Ny:, :] - (sol_rot_dec.reshape(-1, 1)[Nx*Ny:, :] + u_ref[Nx*Ny:, :])) ** 2
            errors_p_den[i, 0] = np.linalg.norm(u_test[Nx*Ny:, i] + initial_state[Nx*Ny:, 0]) ** 2

        err   = np.sqrt(np.sum(errors,   axis=0) / np.sum(errors_den,   axis=0))[0]
        err_q = np.sqrt(np.sum(errors_q, axis=0) / np.sum(errors_q_den, axis=0))[0]
        err_p = np.sqrt(np.sum(errors_p, axis=0) / np.sum(errors_p_den, axis=0))[0]

        print(f"error:   {err}")
        proj_errors.append((p_red, err))

    print("\nProjection errors:", proj_errors)

    if write_csv:
        mu_tag = f"{mu_val:.2f}".replace('.', '')
        out_file = filepaths['AE_results'] / f"proj_error_ae_{ae_name}_mu{mu_tag}.csv"
        with open(out_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y"])
            writer.writerows(proj_errors)
        print(f"Saved CSV to: {out_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute AE projection errors for the wave equation experiment.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument('--ae_name', type=str, required=True, choices=list(AE_REGISTRY.keys()), help='Autoencoder architecture name (determines network class and checkpoint lookup)')
    parser.add_argument('--p_red', type=int, nargs='+', default=[4, 8, 12, 16], metavar='P', help='Reduced dimension(s) to evaluate (default: 4 8 12 16)')
    parser.add_argument('--mu_val', type=float, default=0.6, help='Test parameter value mu (default: 0.6)')
    parser.add_argument('--scaled_data', action=argparse.BooleanOptionalAction, default=True, help='Use scaled data (default: True).')
    parser.add_argument('--rotated', action=argparse.BooleanOptionalAction, default=False, help='Use rotated snapshot data (default: True).')
    parser.add_argument('--visualize', action='store_true', default=False,help='Enable visualization during timestepping (default: False)')
    parser.add_argument('--write_csv', action='store_true', default=False, help='Write projection errors to a CSV file')

    args = parser.parse_args()
    proj_error_AE(ae_name=args.ae_name, p_red_values=args.p_red, mu_val=args.mu_val, scaled_data=args.scaled_data, rotated=args.rotated, visualize=args.visualize, write_csv=args.write_csv)