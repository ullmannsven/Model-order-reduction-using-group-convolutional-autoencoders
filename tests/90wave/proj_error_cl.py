#!/usr/bin/env python

import csv
import numpy as np
import pickle

from pymor.basic import *
import os
from pathlib import Path

from experiment_setup import WaveExperiment, WaveExperimentConfig


def proj_error_cl():
 
    # Configure experiment
    config = WaveExperimentConfig(x_flow=True, nt=500, timestep_factor=1)
    experiment = WaveExperiment(config)
   
    Nx = config.Nx
    Ny = config.Ny

    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir = Path(script_dir)
    rb_dir = os.path.join(script_dir, "CL_results")
    rb_dir = Path(rb_dir)
    os.makedirs(rb_dir, exist_ok=True)
    
    base_dir = os.path.join(script_dir, "snapshots_grid")
    os.makedirs(base_dir, exist_ok=True)

    mu_val = 0.6
    #mu_test = experiment.fom.parameters.parse({'mu': mu_val})
    filepaths = experiment.get_filepath_patterns(script_dir)
    filename = filepaths['snapshots'] / "snapshots_256x256_080_nt_500"
    with open(filename, 'rb') as f:
        arr = pickle.load(f)['snapshots']
    u_test = np.vstack(arr).T

    #NOTE: alternativ: compute u_test with x_flow False fom, but takes a lot longer and is not required for 90 degree rotation
    if not config.x_flow:
        u_test = u_test.reshape(2, Nx, Ny, -1)
        u_test = np.rot90(u_test, k=-1, axes=(1,2)) #rotate countercockwise
        u_test = u_test.reshape(2*Nx*Ny, -1)

    initial_state = experiment.get_initial_state(mu_val=mu_val)

    #this line needs to change when centered
    u_test = u_test

    rb_path = rb_dir / f"reduced_basis_256x256_rbsize_50_nt_500"
    with open(rb_path, "rb") as file:
        reduced_basis = pickle.load(file)

    proj_errors = []
    
    for j in [4, 8, 12, 16]:
        rb = reduced_basis[:j//2]
        rb_tsi = rb.transposed_symplectic_inverse()
        space = NumpyVectorSpace(Nx*Ny)
        from pymor.vectorarrays.block import BlockVectorSpace
        block_space = BlockVectorSpace([space, space])
        u_test_1, u_test_2 = np.split(u_test, [int(u_test.shape[0]/2)], axis=0)
        u_test_pymor = block_space.make_array([space.from_numpy(u_test_1), space.from_numpy(u_test_2)])
        u_proj = rb.lincomb(u_test_pymor.inner(rb_tsi.to_array()).T)

        #if j == 50:
        #    space = NumpyVectorSpace(2*Nx*Ny)
        #    experiment.fom.visualize(space.from_numpy(u_proj.to_numpy()[:, 100] + initial_state[:, 0]))
        #    #experiment.fom.visualize(space.from_numpy(u_test[:, 100] + initial_state[:, 0]))
        #    experiment.fom.visualize(space.from_numpy(u_test[:, 100] + initial_state[:, 0]) - space.from_numpy(u_proj.to_numpy()[:, 100] + initial_state[:, 0]))


        error = np.sum(np.linalg.norm(u_proj.to_numpy().reshape(2*config.Nx*config.Ny, -1) - u_test, axis=0))
        error_den = np.sum(np.linalg.norm(u_test + initial_state, axis=0))

        proj_errors.append((j, np.sqrt(error/error_den)))

    print(proj_errors)

    #save data as csv
    import csv

    # Target folder (relative or absolute)
    out_file = rb_dir / f"proj_error_cl_mu{mu_val}.csv"

    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y"])   # header
        writer.writerows(proj_errors)

if __name__ == '__main__':
    proj_error_cl()    
