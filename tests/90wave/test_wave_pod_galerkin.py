#!/usr/bin/env python
"""
Test POD Galerkin ROM for the wave equation experiment.

Usage:
    python test_wave_pymor_pod.py [--mu_val MU] [--p_red P [P ...]]
                                  [--rb_size N] [--visualize] [--save_data]

Arguments:
    --mu_val    Test parameter value (default: 0.6)
    --p_red     One or more reduced dimensions to evaluate (default: 4 8 12 16)
    --rb_size   Size of the reduced basis to load (default: 50)
    --visualize Enable visualization of ROM solutions (default: False)
    --save_data Save reconstruction errors to a CSV file (default: False)

Examples:
    # Default settings
    python test_wave_pymor_pod.py

    # Custom mu and p_red, save results
    python test_wave_pymor_pod.py --mu_val 0.8 --p_red 4 8 16 --save_data

    # With visualization
    python test_wave_pymor_pod.py --visualize
"""

import argparse
import csv
import numpy as np
import pickle
import os
from pathlib import Path

from pymor.basic import *
from pymor.reductors.basic import InstationaryRBReductor
from pymor.vectorarrays.block import BlockVectorSpace

from experiment_setup import WaveExperimentConfig, WaveExperiment


def test_wave_pymor_pod(mu_val=0.6, p_red_values=[4, 8, 12, 16], rb_size=50, visualize=False, save_data=False):

    config = WaveExperimentConfig(nt=500, timestep_factor=1, visualize_q=visualize)
    experiment = WaveExperiment(config)
    steps = config.nt

    Nx = config.Nx
    Ny = config.Ny

    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    filepaths = experiment.get_filepath_patterns(script_dir)

    mu_tag = f"{mu_val:.2f}".replace('.', '')
    mu_test = experiment.fom.parameters.parse({'mu': mu_val})
    filename = filepaths['snapshots'] / f"snapshots_{Nx}x{Ny}_{mu_tag}_nt_{config.nt}"
    with open(filename, 'rb') as f:
        arr = pickle.load(f)['snapshots']
    u_test = np.vstack(arr).T

    rb_path = filepaths['pod_results'] / f"reduced_basis_uncentered_{Nx}x{Ny}_rbsize_{rb_size}_nt_{config.nt}.npy"
    reduced_basis_all = np.load(rb_path)

    initial_state = experiment.get_initial_state(mu_val=mu_val)
    u_test = u_test + initial_state

    reconstruction_errors = []

    for p_red in p_red_values:
        reduced_basis = reduced_basis_all[:, :p_red]

        space2 = NumpyVectorSpace(reduced_basis.shape[0])
        space = NumpyVectorSpace(reduced_basis.shape[0] // 2)
        block_space = BlockVectorSpace([space, space])
        rb_1, rb_2 = np.split(reduced_basis, [reduced_basis.shape[0] // 2])
        rb = block_space.make_array([space.from_numpy(rb_1), space.from_numpy(rb_2)])

        reductor = InstationaryRBReductor(experiment.fom, rb)
        rom = reductor.reduce()
        u_rom = rom.solve(mu_test)
        u_rec = reductor.reconstruct(u_rom).to_numpy().reshape(2 * Nx * Ny, -1)

        if visualize:
            experiment.fom.visualize(reductor.reconstruct(u_rom)[0])
            experiment.fom.visualize(reductor.reconstruct(u_rom)[50])
            experiment.fom.visualize(reductor.reconstruct(u_rom)[100])
            experiment.fom.visualize(space2.make_array(u_test))

        error = np.sqrt(np.sum(np.linalg.norm(u_test[:, :steps] - u_rec[:, :steps], axis=0) ** 2))
        error_den = np.sqrt(np.sum(np.linalg.norm(u_test[:, :steps]) ** 2))

        error_q = np.sqrt(np.sum(np.linalg.norm(u_test[:Nx*Ny, :steps] - u_rec[:Nx*Ny, :steps], axis=0) ** 2))
        error_den_q = np.sqrt(np.sum(np.linalg.norm(u_test[:Nx*Ny, :steps]) ** 2))

        error_p = np.sqrt(np.sum(np.linalg.norm(u_test[Nx*Ny:, :steps] - u_rec[Nx*Ny:, :steps], axis=0) ** 2))
        error_den_p = np.sqrt(np.sum(np.linalg.norm(u_test[Nx*Ny:, :steps]) ** 2))

        print(f"p_red={p_red}: error={error/error_den:.4e}, error_q={error_q/error_den_q:.4e}, error_p={error_p/error_den_p:.4e}")
        reconstruction_errors.append((p_red, error / error_den))

    print(reconstruction_errors)

    if save_data:
        out_file = filepaths['pod_results'] / f"reconstruction_error_pymor_pod_mu{mu_val}.csv"
        with open(out_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y"])
            writer.writerows(reconstruction_errors)
        print(f"Saved CSV to: {out_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test POD Galerkin ROM for the wave equation experiment.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--mu_val', type=float, default=0.6, help='Test parameter value mu (default: 0.6)')
    parser.add_argument('--p_red', type=int, nargs='+', default=[4, 8, 12, 16], metavar='P', help='Reduced dimension(s) to evaluate (default: 4 8 12 16)')
    parser.add_argument('--rb_size', type=int, default=50, help='Size of the reduced basis to load (default: 50)')
    parser.add_argument('--visualize', action='store_true', default=False, help='Enable visualization of ROM solutions (default: False)')
    parser.add_argument('--save_data', action='store_true', default=False, help='Save reconstruction errors to a CSV file (default: False)')

    args = parser.parse_args()
    test_wave_pymor_pod(mu_val=args.mu_val, p_red_values=args.p_red, rb_size=args.rb_size, visualize=args.visualize, save_data=args.save_data)