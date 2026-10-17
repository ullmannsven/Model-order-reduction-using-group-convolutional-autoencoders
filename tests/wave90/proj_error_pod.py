#!/usr/bin/env python
"""
Compute POD projection errors for the wave equation experiment.

Usage:
    python proj_error_pod.py [--mu_val MU] [--p_red P [P ...]]
                             [--rb_size N] [--centered] [--visualize] [--write_csv]

Arguments:
    --mu_val    Test parameter value (default: 0.6)
    --p_red     One or more reduced dimensions to evaluate (default: 4 8 12 16)
    --rb_size   Size of the reduced basis to load (default: 50)
    --centered  Load the centered reduced basis. Default: uncentered.
    --visualize Enable visualization during timestepping (default: False).
    --write_csv Write projection errors to a CSV file.

Examples:
    # Default settings
    python proj_error_pod.py

    # Custom mu and p_red values
    python proj_error_pod.py --mu_val 0.8 --p_red 4 8 16

    # Centered basis, write CSV
    python proj_error_pod.py --centered --write_csv
"""

import argparse
import csv
import numpy as np
import pickle
import os
from pathlib import Path

from pymor.basic import *

from experiment_setup import WaveExperiment, WaveExperimentConfig


def proj_error_pod(mu_val=0.6, p_red_values=[4, 8, 12, 16], rb_size=50, centered=False, visualize=False, write_csv=False):

    config = WaveExperimentConfig(x_flow=True, nt=500, timestep_factor=1)
    experiment = WaveExperiment(config)

    Nx = config.Nx
    Ny = config.Ny
    timestep_factor = config.timestep_factor

    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    filepaths = experiment.get_filepath_patterns(script_dir)

    if centered:
        rb_path = filepaths['pod_results'] / f"reduced_basis_{Nx}x{Ny}_rbsize_{rb_size}_nt_{config.nt}.npy"
    else:
        rb_path = filepaths['pod_results'] / f"reduced_basis_uncentered_{Nx}x{Ny}_rbsize_{rb_size}_nt_{config.nt}.npy"
    
    reduced_basis_all = np.load(rb_path)

    mu_tag = f"{mu_val:.2f}".replace('.', '')
    filename = filepaths['snapshots'] / f"snapshots_{Nx}x{Ny}_{mu_tag}_nt_{config.nt}"
    with open(filename, 'rb') as f:
        arr = pickle.load(f)['snapshots']
    u_test = np.vstack(arr).T

    if not config.x_flow:
        u_test = u_test.reshape(2, Nx, Ny, -1)
        u_test = np.rot90(u_test, k=-1, axes=(1, 2))
        u_test = u_test.reshape(2 * Nx * Ny, -1)

    initial_state = experiment.get_initial_state(mu_val=mu_val)

    amount_of_iters = int(config.nt / timestep_factor)
    errors = np.zeros((amount_of_iters, 1))
    errors_den = np.zeros((amount_of_iters, 1))

    proj_errors = []

    for p_red in p_red_values:
        reduced_basis = reduced_basis_all[:, :p_red]

        for i in range(amount_of_iters):
            if centered:
                sol = u_test[:, i]
            else:
                sol = u_test[:, i] + initial_state[:, 0]

            sol_enc = reduced_basis.T @ sol
            sol_dec = reduced_basis @ sol_enc

            if visualize and i == 100:
                space2 = NumpyVectorSpace(config.Nx * config.Ny * 2)
                experiment.fom.visualize(space2.from_numpy(sol.reshape(-1, 1)))
                experiment.fom.visualize(space2.from_numpy((reduced_basis @ (reduced_basis.T @ sol)).reshape(-1, 1)))
                experiment.fom.visualize(space2.from_numpy((sol - reduced_basis @ (reduced_basis.T @ sol)).reshape(-1, 1)))

            errors[i, 0] = np.linalg.norm(sol_dec.reshape(-1, 1) - sol.reshape(-1, 1)) ** 2
            errors_den[i, 0] = np.linalg.norm(u_test[:, i] + initial_state[:, 0]) ** 2

        proj_errors.append((p_red, np.sqrt(np.sum(errors, axis=0) / np.sum(errors_den, axis=0))[0]))

    print(proj_errors)

    if write_csv:
        out_file = filepaths['pod_results'] / f"proj_error_pod_mu{mu_tag}.csv"
        with open(out_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y"])
            writer.writerows(proj_errors)
        print(f"Saved CSV to: {out_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute POD projection errors for the wave equation experiment.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--mu_val', type=float, default=0.6, help='Test parameter value mu (default: 0.6)')
    parser.add_argument('--p_red',type=int,nargs='+',default=[4, 8, 12, 16],metavar='P',help='Reduced dimension(s) to evaluate (default: 4 8 12 16)')
    parser.add_argument('--rb_size',type=int,default=50,help='Size of the reduced basis to load (default: 50)')
    parser.add_argument('--centered',action='store_true',default=False,help='Load the centered reduced basis. Default: uncentered.')
    parser.add_argument('--visualize', action='store_true', default=False, help='Enable visualization during timestepping (default: False).')
    parser.add_argument('--write_csv',action='store_true',default=False,help='Write projection errors to a CSV file.')

    args = parser.parse_args()
    proj_error_pod(mu_val=args.mu_val, p_red_values=args.p_red, rb_size=args.rb_size, centered=args.centered, visualize=args.visualize, write_csv=args.write_csv)