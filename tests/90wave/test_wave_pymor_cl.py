#!/usr/bin/env python

import time
import numpy as np
import pickle
import os
from pathlib import Path

from pymor.basic import *
from pymor.reductors.symplectic import QuadraticHamiltonianRBReductor
from experiment_setup import WaveExperimentConfig, WaveExperiment

def test_wave_CL_galerkin():

    # Configure experiment
    config = WaveExperimentConfig(x_flow=True, nt=500, timestep_factor=1, visualize_q=False)
    experiment = WaveExperiment(config)

    save_data = True
    visualize = False
    steps = 100

    Nx = config.Nx
    Ny = config.Ny

    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir = Path(script_dir)
    filepaths = experiment.get_filepath_patterns(script_dir)

    # Load test data
    mu_val = 0.6
    mu_val_tag = f"{mu_val:.2f}".replace('.', '')
    mu_test = experiment.fom.parameters.parse({'mu': mu_val})
    filename = filepaths['snapshots'] / f"snapshots_{Nx}x{Ny}_{mu_val_tag}_nt_{config.nt}"
    with open(filename, 'rb') as f:
        arr = pickle.load(f)['snapshots']
    u_test = np.vstack(arr).T

    q0, p0 = experiment._get_initial_condition(mu_val=mu_val)
    initial_state = np.hstack((q0, p0)).reshape(-1, 1)
    u_test = u_test + initial_state

    
    rb_path = filepaths['cl_results'] / f"reduced_basis_uncentered_{Nx}x{Ny}_rbsize_50_nt_{config.nt}"
    with open(rb_path, 'rb') as file:
        reduced_basis = pickle.load(file)

    reconstruction_errors = []

    for p_red in [4, 8, 12, 16]:


        RB = reduced_basis[:p_red//2]
        reductor = QuadraticHamiltonianRBReductor(experiment.fom, RB)
        rom = reductor.reduce()
        u_rom = rom.solve(mu=mu_test)

        if visualize:
            pass
            #space2 = NumpyVectorSpace(2*Nx*Ny)   
            #experiment.fom.visualize(reductor.reconstruct(u_rom)[0])
            #experiment.fom.visualize(reductor.reconstruct(u_rom)[50])
            #experiment.fom.visualize(reductor.reconstruct(u_rom)[100])
            #experiment.fom.visualize(space2.make_array(u_test[:, 100]))

        
        error = np.sqrt(np.sum(np.linalg.norm(u_test[:, :steps] - reductor.reconstruct(u_rom).to_numpy().reshape(2*Nx*Ny, -1)[:, :steps], axis=0)**2))
        error_den = np.sqrt(np.sum(np.linalg.norm(u_test[:, :steps])**2))

        error_q = np.sqrt(np.sum(np.linalg.norm(u_test[:Nx*Ny, :steps] - reductor.reconstruct(u_rom).to_numpy().reshape(2*Nx*Ny, -1)[:Nx*Ny, :steps], axis=0)**2))
        error_den_q = np.sqrt(np.sum(np.linalg.norm(u_test[:Nx*Ny, :steps])**2))

        error_p = np.sqrt(np.sum(np.linalg.norm(u_test[Nx*Ny:, :steps] - reductor.reconstruct(u_rom).to_numpy().reshape(2*Nx*Ny, -1)[Nx*Ny:, :steps], axis=0)**2))
        error_den_p = np.sqrt(np.sum(np.linalg.norm(u_test[Nx*Ny:, :steps])**2))

        reconstruction_errors.append((p_red, error / error_den))

    print(reconstruction_errors)

    if save_data:
        import csv
        out_file = filepaths['cl_results'] / f"reconstruction_error_pymor_cl_mu{mu_val}.csv"

        with open(out_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y"])   # header
            writer.writerows(reconstruction_errors)


if __name__ == '__main__':
    test_wave_CL_galerkin()
