#!/usr/bin/env python
"""
Compute projection errors for a trained Autoencoder on the wave equation experiment.

Usage:
    python proj_error_AE.py [--ae_name AE_NAME] [--xflow] [--p_red P [P ...]] [--timestamp TIMESTAMP] [--mu_val MU] [--scaled_data] [--visualize] [--write_csv]

Arguments:
    --ae_name       Name of the autoencoder architecture. Determines which network
                    class is used and how checkpoint files are located.
                    Choices:
                        RotationUpsamplingGCNN_C4   -> RotationUpsamplingGCNNAutoencoder2D (N=4)
                        RotationUpsamplingGCNN_C8   -> RotationUpsamplingGCNNAutoencoder2D (N=8)
                        UpsamplingCNN               -> UpsamplingCNNAutoencoder2D
                        TrivialUpsamplingGCNN       -> TrivialUpsamplingGCNNAutoencoder2D
                        InvariantPoseGCNN_C4        -> InvariantPoseGCNNAutoencoder2D (N=4)
                        InvariantPoseGCNN_C8        -> InvariantPoseGCNNAutoencoder2D (N=8)
                    For the invariant/pose autoencoders the decoder uses the known pose of the test
                    data (identity for x-flow, rotation by -90 degrees for y-flow); the pose estimated
                    by the network is only reported as a diagnostic.
    --xflow         Enable xflow (default: True).
    --p_red         One or more reduced dimensions to evaluate (default: 4 8 12 16)
    --timestamp     Optional saved-run timestamp; appends _t_TIMESTAMP to both input filenames.
    --mu_val        Test parameter value (default: 0.8)
    --scaled_data   Use scaled data (default: True)
    --visualize     Enable visualization during timestepping (default: False)
    --write_csv     Write projection errors to a CSV file (default: False)


Examples:
    # C8 equivariant network, default p_red values
    python proj_error_AE.py --ae_name RotationUpsamplingGCNN_C8 --timestamp 09_02_2026-11_14_30

    # CNN baseline, single p_red, write CSV
    python proj_error_AE.py --ae_name UpsamplingCNN --p_red 8 --write_csv 

    # C4 network, multiple p_red values, different mu, visualize 
    python proj_error_AE.py --ae_name RotationUpsamplingGCNN_C4 --p_red 4 8 16 --mu_val 0.75 --visualize

    # yflow 
    python proj_error_AE.py --ae_name RotationUpsamplingGCNN_C4 --no-xflow --p_red 4

    # invariant/pose autoencoder on the rotated (y-flow) problem
    python proj_error_AE.py --ae_name InvariantPoseGCNN_C4 --no-xflow --p_red 12
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

from equiv_networks.autoencoders import RotationUpsamplingGCNNAutoencoder2D, UpsamplingCNNAutoencoder2D, InvariantPoseGCNNAutoencoder2D
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
    'UpsamplingCNN': {
        'class': UpsamplingCNNAutoencoder2D,
        'gspace': None,
    },
    'UpsamplingCNN_Symplectic': {
        'class': UpsamplingCNNAutoencoder2D,
        'gspace': None,
    },
    'RotationUpsamplingGCNN_bothdir': {
        'class': RotationUpsamplingGCNNAutoencoder2D,
        'gspace': lambda: gspaces.rot2dOnR2(N=4),
    }, 
    'UpsamplingCNN_bothdir': {
        'class': UpsamplingCNNAutoencoder2D,
        'gspace': None,
    },
    'InvariantPoseGCNN_C4': {
        'class': InvariantPoseGCNNAutoencoder2D,
        'gspace': lambda: gspaces.rot2dOnR2(N=4),
    },
    'InvariantPoseGCNN_C8': {
        'class': InvariantPoseGCNNAutoencoder2D,
        'gspace': lambda: gspaces.rot2dOnR2(N=8),
    },
}


def known_pose(group_order, x_flow):
    """Pose index k (element g_k of C_N) of the test data relative to the training orientation (x-flow).

    The y-flow test snapshots are generated as np.rot90(u, k=-1, axes=(1, 2)), i.e. torch.rot90(x, -1) on the
    (row, column) axes. In escnn's convention the generator of C4 acts as torch.rot90(x, 1), so this is the
    element g_{3N/4}: pose 3 for C4 and pose 6 for C8.
    """
    if x_flow:
        return 0
    return (3 * group_order // 4) % group_order


def proj_error_AE(ae_name, xflow, p_red_values, mu_val= 0.8, scaled_data = True, visualize=False, write_csv = False, timestamp=None):

    config = WaveExperimentConfig(x_flow=xflow, visualize_q=True, nt=500, timestep_factor=1)
    experiment = WaveExperiment(config)

    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    filepaths = experiment.get_filepath_patterns(script_dir)

    Nx = config.Nx
    Ny = config.Ny
    timestep_factor = config.timestep_factor
    grid = f"{Nx}x{Ny}"

    checkpoint_dir = script_dir / "checkpoints"

    # Retrieve AE entry from registry
    ae_entry = AE_REGISTRY[ae_name]
    network_class = ae_entry['class']

    proj_errors = []

    for p_red in p_red_values:
        print(f"\n--- p_red = {p_red} ---")
        stem = f"wave_2D_{ae_name}_p_{p_red}_{grid}"
        if timestamp:
            stem += f"_t_{timestamp}"
        nn_save_filepath = checkpoint_dir / f"{stem}.pt"
        network_parameters_file = script_dir / "network_parameters" / f"{stem}.pkl"

        with Path(network_parameters_file).open("rb") as f:
            parameters = pickle.load(f)

        scaler = Scaler(dims=config.dims)

        # Inject gspace into network parameters if required
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
        device = next(model.network.parameters()).device

        # Invariant/pose autoencoders do not store the orientation in the latent code: the decoder is told the
        # (known) rotation of the test problem once. This must happen before compute_reference_offset, since
        # u_ref = u_0 - decode(encode(0)) uses the same decoder.
        uses_pose = hasattr(model.network, "set_pose")
        pose_agreement = []
        if uses_pose:
            pose = known_pose(model.network.group_order, config.x_flow)
            model.network.set_pose(pose)
            print(f"Decoding with known pose {pose} of C{model.network.group_order}")

        #mu_tag = f"{mu_val:.2f}".replace('.', '')
        filename = filepaths['snapshots'] / f"snapshots_{grid}_{mu_val}_nt_{config.nt}"
        with open(filename, 'rb') as f:
            arr = pickle.load(f)['snapshots']
        u_test = np.vstack(arr).T

        if not config.x_flow:
            u_test = u_test.reshape(2, Nx, Ny, -1)
            u_test = np.rot90(u_test, k=-1, axes=(1, 2))
            u_test = u_test.reshape(2 * Nx * Ny, -1)

        initial_state = experiment.get_initial_state(mu_val=mu_val)
        u_ref, _ = experiment.compute_reference_offset(model, mu_val=mu_val, scaled_data=scaled_data)
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
                net_input = torch.as_tensor(scaler.scale(scaler.restrict(sol_rot)), dtype=torch.float32, device=device).unsqueeze(0)
            else:
                net_input = torch.as_tensor(scaler.restrict(sol_rot), dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                if uses_pose:
                    # invariant code; the estimated pose is only compared against the known pose
                    sol_rot_enc, estimated_pose = model.network.encode_with_pose(net_input)
                    # the first snapshot is the zero state (constant field after scaling): its pose is undefined
                    if i > 0:
                        pose_agreement.append(int(estimated_pose[0]) == pose)
                else:
                    sol_rot_enc = model.network.encode(net_input)
                # for invariant/pose autoencoders, decode() applies the known pose set above
                sol_rot_dec = model.network.decode(sol_rot_enc)[0].cpu().numpy()

            if scaled_data:
                sol_rot_dec = scaler.prolongate(scaler.unscale(sol_rot_dec))
            else:
                sol_rot_dec = scaler.prolongate(sol_rot_dec)

            if visualize and i == 100:
                space2 = NumpyVectorSpace(config.Nx * config.Ny * 2)
                experiment.fom.visualize(space2.from_numpy((sol_rot_dec + u_ref.reshape(-1, 1))))
                experiment.fom.visualize(space2.from_numpy(sol_rot.reshape(-1,1) + initial_state))
                experiment.fom.visualize(space2.from_numpy(sol_rot.reshape(-1,1) + initial_state - sol_rot_dec - u_ref.reshape(-1, 1)))

            errors[i, 0] = np.linalg.norm(sol_rot.reshape(-1, 1) - sol_rot_dec.reshape(-1, 1)) ** 2
            errors_den[i, 0]= np.linalg.norm(u_test[:, i]) ** 2

            errors_q[i, 0] = np.linalg.norm(sol_rot.reshape(-1, 1)[:Nx*Ny, :] + initial_state[:Nx*Ny, :]- (sol_rot_dec.reshape(-1, 1)[:Nx*Ny, :] + u_ref[:Nx*Ny, :])) ** 2
            errors_q_den[i, 0] = np.linalg.norm(u_test[:Nx*Ny, i] + initial_state[:Nx*Ny, 0]) ** 2

            errors_p[i, 0] = np.linalg.norm(sol_rot.reshape(-1, 1)[Nx*Ny:, :] + initial_state[Nx*Ny:, :]- (sol_rot_dec.reshape(-1, 1)[Nx*Ny:, :] + u_ref[Nx*Ny:, :])) ** 2
            errors_p_den[i, 0] = np.linalg.norm(u_test[Nx*Ny:, i] + initial_state[Nx*Ny:, 0]) ** 2

        err = np.sqrt(np.sum(errors,   axis=0) / np.sum(errors_den,   axis=0))[0]
        err_q = np.sqrt(np.sum(errors_q, axis=0) / np.sum(errors_q_den, axis=0))[0]
        err_p = np.sqrt(np.sum(errors_p, axis=0) / np.sum(errors_p_den, axis=0))[0]

        proj_errors.append((p_red, err))

        if uses_pose:
            print(f"Estimated pose equals known pose for {100 * np.mean(pose_agreement):.1f}% of the snapshots")

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
    parser.add_argument('--xflow', action=argparse.BooleanOptionalAction, default=True, help='Enable xflow (default: True).')
    parser.add_argument('--p_red', type=int, nargs='+', default=[4, 8, 12, 16], metavar='P', help='Reduced dimension(s) to evaluate (default: 4 8 12 16)')
    parser.add_argument( '--mu_val', type=float, default=0.8, help='Test parameter value mu (default: 0.8)')
    parser.add_argument('--scaled_data', action=argparse.BooleanOptionalAction, default=True, help='Use scaled data (default: True).')
    parser.add_argument('--visualize', action='store_true', default=False, help='Enable visualization during timestepping (default: False)')
    parser.add_argument('--write_csv', action='store_true', default=False, help='Write projection errors to a CSV file')
    
    parser.add_argument('--timestamp', type=str, default=None, help='Saved-run timestamp (e.g. 09_02_2026-11_14_30); appends _t_TIMESTAMP to checkpoint and network-parameter filenames')

    args = parser.parse_args()
    proj_error_AE(ae_name=args.ae_name, xflow=args.xflow, p_red_values=args.p_red, mu_val=args.mu_val, scaled_data=args.scaled_data, visualize=args.visualize, write_csv=args.write_csv, timestamp=args.timestamp)