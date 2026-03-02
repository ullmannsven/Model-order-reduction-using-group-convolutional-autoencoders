#!/usr/bin/env python

import time
import numpy as np
import pickle
import os
from pathlib import Path

from pymor.basic import *
from experiment_setup import WaveExperimentConfig, WaveExperiment

from pymor.reductors.basic import InstationaryRBReductor
from pymor.vectorarrays.block import BlockVectorSpace

def test_wave_pymor_pod(): 

    # Configure experiment
    config = WaveExperimentConfig(nt=500, timestep_factor=1, visualize_q=False)
    experiment = WaveExperiment(config)

    save_data = True
    visualize = False
    steps = 100

    script_dir = os.path.dirname(os.path.abspath(__file__))
    rb_dir = os.path.join(script_dir, "pod_results")
    script_dir = Path(script_dir)
    rb_dir = Path(rb_dir)
    os.makedirs(rb_dir, exist_ok=True)

    filepaths = experiment.get_filepath_patterns(script_dir)
    
    # Load test data: Using precomputed snapshots for speed.
    mu_test_val = 0.6
    mu_test = experiment.fom.parameters.parse({'mu': mu_test_val})
    filename = filepaths['snapshots'] / "snapshots_256x256_060_nt_500"
    with open(filename, 'rb') as f:
        arr = pickle.load(f)['snapshots']
    u_test = np.vstack(arr).T

    Nx = config.Nx
    Ny = config.Ny

    rb_path = rb_dir / f"reduced_basis_uncentered_{config.Nx}x{config.Ny}_rbsize_50_nt_{config.nt}.npy"
    reduced_basis_all = np.load(rb_path)

    #this needs to be done as we are loading data right now: 
    initial_state = experiment.get_initial_state(mu_val=mu_test_val)
    u_test = u_test + initial_state

    reconstruction_errors = []

    for p_red in [4,8,12,16]:
        
        reduced_basis = reduced_basis_all[:, :p_red]

        space2 = NumpyVectorSpace(reduced_basis.shape[0])
        space = NumpyVectorSpace(int(reduced_basis.shape[0] / 2))
        block_space = BlockVectorSpace([space, space])
        rb_1, rb_2 = np.split(reduced_basis, [int(reduced_basis.shape[0] / 2)])
        rb = block_space.make_array([space.from_numpy(rb_1), space.from_numpy(rb_2)])

        reductor = InstationaryRBReductor(experiment.fom, rb)
        rom = reductor.reduce()
        u_rom = rom.solve(mu_test)

        if visualize:
            pass
            # + block_space.make_array([space.from_numpy(q0), space.from_numpy(p0)]
            #experiment.fom.visualize(reductor.reconstruct(u_rom)[0])
            #experiment.fom.visualize(reductor.reconstruct(u_rom)[50])
            #experiment.fom.visualize(reductor.reconstruct(u_rom)[100])
            #experiment.fom.visualize(space2.make_array(u_test))

        error = np.sqrt(np.sum(np.linalg.norm(u_test[:, :steps] - reductor.reconstruct(u_rom).to_numpy().reshape(2*Nx*Ny, -1)[:, :steps], axis=0)**2))
        error_den = np.sqrt(np.sum(np.linalg.norm(u_test[:, :steps])**2))

        error_q = np.sqrt(np.sum(np.linalg.norm(u_test[:Nx*Ny, :steps] - reductor.reconstruct(u_rom).to_numpy().reshape(2*Nx*Ny, -1)[:Nx*Ny, :steps], axis=0)**2))
        error_den_q = np.sqrt(np.sum(np.linalg.norm(u_test[:Nx*Ny, :steps])**2))

        error_p = np.sqrt(np.sum(np.linalg.norm(u_test[Nx*Ny:, :steps] - reductor.reconstruct(u_rom).to_numpy().reshape(2*Nx*Ny, -1)[Nx*Ny:, :steps], axis=0)**2))
        error_den_p = np.sqrt(np.sum(np.linalg.norm(u_test[Nx*Ny:, :steps])**2))

        #print("was ist der Fehler", error/error_den)
        #print("was ist der Fehler in q", error_q/error_den_q)
        #print("was ist der Fehler in p", error_p/error_den_p)

        reconstruction_errors.append((p_red, error/error_den))

    print(reconstruction_errors)

    if save_data:
        #save data as csv
        import csv

        # Target folder (relative or absolute)
        out_file = rb_dir / f"reconstruction_error_pymor_pod_mu{mu_test_val}.csv"

        with open(out_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y"])   # header
            writer.writerows(reconstruction_errors)

if __name__ == '__main__':
    test_wave_pymor_pod()
