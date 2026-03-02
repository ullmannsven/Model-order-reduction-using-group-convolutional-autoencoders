#!/usr/bin/env python
"""
Compute CL projection errors for the wave equation experiment.

Usage:
    python proj_error_cl.py [--mu_val MU] [--p_red P [P ...]]
                            [--rb_size N] [--centered] [--write_csv]

Arguments:
    --mu_val    Test parameter value (default: 0.6)
    --p_red     One or more reduced dimensions to evaluate (default: 4 8 12 16)
    --rb_size   Size of the reduced basis to load (default: 50)
    --centered  Load the centered reduced basis. Default: uncentered.
    --write_csv Write projection errors to a CSV file.

Examples:
    # Default settings
    python proj_error_cl.py

    # Custom mu and p_red values
    python proj_error_cl.py --mu_val 0.8 --p_red 4 8 16

    # Centered basis, write CSV
    python proj_error_cl.py --centered --write_csv
"""

import argparse
import csv
import numpy as np
import pickle
import os
from pathlib import Path

from pymor.basic import *
from pymor.vectorarrays.block import BlockVectorSpace

from experiment_setup import WaveExperiment, WaveExperimentConfig


def proj_error_cl(mu_val=0.6, p_red_values=[4, 8, 12, 16], rb_size=50, centered=False, write_csv=False):

    config = WaveExperimentConfig(x_flow=True, nt=500, timestep_factor=1)
    experiment = WaveExperiment(config)

    Nx = config.Nx
    Ny = config.Ny

    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    filepaths = experiment.get_filepath_patterns(script_dir)

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

    if centered:
        rb_path = filepaths['cl_results'] / f"reduced_basis_{Nx}x{Ny}_rbsize_{rb_size}_nt_{config.nt}"
    else:
        rb_path = filepaths['cl_results'] / f"reduced_basis_uncentered_{Nx}x{Ny}_rbsize_{rb_size}_nt_{config.nt}"

    with open(rb_path, "rb") as file:
        reduced_basis = pickle.load(file)

    proj_errors = []

    space = NumpyVectorSpace(Nx * Ny)
    block_space = BlockVectorSpace([space, space])
    u_test_1, u_test_2 = np.split(u_test, [u_test.shape[0] // 2], axis=0)
    u_test_pymor = block_space.make_array([space.from_numpy(u_test_1), space.from_numpy(u_test_2)])

    for p_red in p_red_values:
        rb = reduced_basis[:p_red // 2]
        rb_tsi = rb.transposed_symplectic_inverse()
        u_proj = rb.lincomb(u_test_pymor.inner(rb_tsi.to_array()).T)

        error = np.sum(np.linalg.norm(u_proj.to_numpy().reshape(2 * Nx * Ny, -1) - u_test, axis=0))
        error_den = np.sum(np.linalg.norm(u_test + initial_state, axis=0))

        proj_errors.append((p_red, np.sqrt(error / error_den)))

    print(proj_errors)

    if write_csv:
        out_file = filepaths['cl_results'] / f"proj_error_cl_mu{mu_tag}.csv"
        with open(out_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y"])
            writer.writerows(proj_errors)
        print(f"Saved CSV to: {out_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute CL projection errors for the wave equation experiment.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--mu_val',
        type=float,
        default=0.6,
        help='Test parameter value mu (default: 0.6)',
    )
    parser.add_argument(
        '--p_red',
        type=int,
        nargs='+',
        default=[4, 8, 12, 16],
        metavar='P',
        help='Reduced dimension(s) to evaluate (default: 4 8 12 16)',
    )
    parser.add_argument(
        '--rb_size',
        type=int,
        default=50,
        help='Size of the reduced basis to load (default: 50)',
    )
    parser.add_argument(
        '--centered',
        action='store_true',
        default=False,
        help='Load the centered reduced basis. Default: uncentered.',
    )
    parser.add_argument(
        '--write_csv',
        action='store_true',
        default=False,
        help='Write projection errors to a CSV file.',
    )

    args = parser.parse_args()
    proj_error_cl(mu_val=args.mu_val, p_red_values=args.p_red, rb_size=args.rb_size, centered=args.centered, write_csv=args.write_csv)