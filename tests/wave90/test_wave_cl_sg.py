#!/usr/bin/env python
"""
Test CL Galerkin ROM for the wave equation experiment. Uses uncentered data, as this is more convient for the pymor reductor.

Usage:
    python test_wave_cl_sg.py [--mu_val MU] [--p_red P [P ...]]
                                    [--rb_size N] [--visualize] [--save_data]

Arguments:
    --mu_val    Test parameter value (default: 0.6)
    --p_red     One or more reduced dimensions to evaluate (default: 4 8 12 16)
    --rb_size   Size of the reduced basis to load (default: 50)
    --visualize Enable visualization of ROM solution (default: False)
    --save_data Save reduction errors to a CSV file (dafuult: False)

Examples:
    # Default settings
    python test_wave_cl_sg.py

    # Custom mu and p_red, save results
    python test_wave_cl_sg.py --mu_val 0.8 --p_red 4 8 16 --save_data

    # With visualization
    python test_wave_cl_sg.py --visualize
"""

import argparse
import csv
import numpy as np
import pickle
import os
from pathlib import Path

from pymor.basic import *
from pymor.reductors.symplectic import QuadraticHamiltonianRBReductor
from experiment_setup import WaveExperimentConfig, WaveExperiment


def test_wave_cl_sg(mu_val=0.6, p_red_values=[4, 8, 12, 16], rb_size=50, visualize=False, save_data=False):

    config = WaveExperimentConfig(x_flow=True, nt=500, timestep_factor=1, visualize_q=visualize)
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

    q0, p0 = experiment._get_initial_condition(mu_val=mu_val)
    initial_state = np.hstack((q0, p0)).reshape(-1, 1)
    u_test = u_test + initial_state

    rb_path = filepaths['cl_results'] / f"reduced_basis_uncentered_{Nx}x{Ny}_rbsize_{rb_size}_nt_{config.nt}"
    with open(rb_path, 'rb') as file:
        reduced_basis = pickle.load(file)

    reduction_errors = []

    for p_red in p_red_values:
        RB = reduced_basis[:p_red // 2]
        reductor = QuadraticHamiltonianRBReductor(experiment.fom, RB)
        rom = reductor.reduce()
        u_rom = rom.solve(mu=mu_test)
        u_rec = reductor.reconstruct(u_rom).to_numpy().reshape(2 * Nx * Ny, -1)

        if visualize:
            space2 = NumpyVectorSpace(2 * Nx * Ny)
            experiment.fom.visualize(reductor.reconstruct(u_rom)[0])
            experiment.fom.visualize(reductor.reconstruct(u_rom)[50])
            experiment.fom.visualize(reductor.reconstruct(u_rom)[100])
            experiment.fom.visualize(space2.make_array(u_test[:, 100]))

        error = np.sqrt(np.sum(np.linalg.norm(u_test[:, :steps] - u_rec[:, :steps], axis=0) ** 2))
        error_den = np.sqrt(np.sum(np.linalg.norm(u_test[:, :steps]) ** 2))

        error_q = np.sqrt(np.sum(np.linalg.norm(u_test[:Nx*Ny, :steps] - u_rec[:Nx*Ny, :steps], axis=0) ** 2))
        error_den_q = np.sqrt(np.sum(np.linalg.norm(u_test[:Nx*Ny, :steps]) ** 2))

        error_p = np.sqrt(np.sum(np.linalg.norm(u_test[Nx*Ny:, :steps] - u_rec[Nx*Ny:, :steps], axis=0) ** 2))
        error_den_p = np.sqrt(np.sum(np.linalg.norm(u_test[Nx*Ny:, :steps]) ** 2))

        print(f"p_red={p_red}: error={error/error_den:.4e}, error_q={error_q/error_den_q:.4e}, error_p={error_p/error_den_p:.4e}")
        reduction_errors.append((p_red, error / error_den))

    print(reduction_errors)

    if save_data:
        out_file = filepaths['cl_results'] / f"reduction_error_pymor_cl_mu{mu_val}.csv"
        with open(out_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y"])
            writer.writerows(reduction_errors)
        print(f"Saved CSV to: {out_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test CL Galerkin ROM for the wave equation experiment.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--mu_val', type=float, default=0.6, help='Test parameter value mu (default: 0.6)')
    parser.add_argument('--p_red', type=int, nargs='+', default=[4, 8, 12, 16], metavar='P', help='Reduced dimension(s) to evaluate (default: 4 8 12 16)')
    parser.add_argument('--rb_size', type=int, default=50, help='Size of the reduced basis to load (default: 50)')
    parser.add_argument('--visualize', action='store_true', default=False, help='Enable visualization of ROM solutions.')
    parser.add_argument('--save_data', action='store_true', default=False, help='Save reduction errors to a CSV file.')

    args = parser.parse_args()
    test_wave_cl_sg(mu_val=args.mu_val, p_red_values=args.p_red, rb_size=args.rb_size, visualize=args.visualize, save_data=args.save_data)