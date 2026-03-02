#!/usr/bin/env python

from xml.parsers.expat import model
import numpy as np
import pickle

from pymor.basic import *
import os
from pathlib import Path
import csv

from experiment_setup import WaveExperiment, WaveExperimentConfig


def proj_error_pod():
 
    # Configure experiment
    config = WaveExperimentConfig(x_flow=True, nt=500, timestep_factor=1)
    experiment = WaveExperiment(config)
    centered = True
   
    timestep_factor = config.timestep_factor
    Nx = config.Nx
    Ny = config.Ny

    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir = Path(script_dir)
    filepaths = experiment.get_filepath_patterns(script_dir)
    
    if centered:
        rb_path = filepaths['pod_results'] / f"reduced_basis_{config.Nx}x{config.Ny}_rbsize_50_nt_{config.nt}.npy"
        reduced_basis_all = np.load(rb_path)
    else:
        rb_path = filepaths['pod_results'] / f"reduced_basis_uncentered_{config.Nx}x{config.Ny}_rbsize_50_nt_{config.nt}.npy"
        reduced_basis_all = np.load(rb_path)
   
    proj_errors = []

    mu_val = 0.6
    mu_val_tag = f"{mu_val:.2f}".replace('.', '')
    filepaths = experiment.get_filepath_patterns(script_dir)
    filename = filepaths['snapshots'] / f"snapshots_{Nx}x{Ny}_{mu_val_tag}_nt_{config.nt}"
    with open(filename, 'rb') as f:
        arr = pickle.load(f)['snapshots']
    u_test = np.vstack(arr).T

    #NOTE: alternativ: compute u_test with x_flow False fom, but takes a lot longer and is not required for 90 degree rotation
    if not config.x_flow:
        u_test = u_test.reshape(2, Nx, Ny, -1)
        u_test = np.rot90(u_test, k=-1, axes=(1,2)) #rotate countercockwise
        u_test = u_test.reshape(2*Nx*Ny, -1)

    initial_state = experiment.get_initial_state(mu_val=mu_val)

    # compute projection error for different RB sizes
    amount_of_iters = int(config.nt/timestep_factor)
    errors = np.zeros((amount_of_iters, 1))
    errors_den = np.zeros((amount_of_iters, 1))
    
    for j in [4, 8, 12, 16]:
        reduced_basis = reduced_basis_all[:, :j]
       
        for i in range(amount_of_iters):
            if centered:
                sol_rot_unscaled = u_test[:, i]
            else: 
                sol_rot_unscaled = u_test[:, i] + initial_state[:, 0]

            sol_rot_unscaled_enc = reduced_basis.T @ sol_rot_unscaled
            sol_rot_unscaled_dec = reduced_basis @ sol_rot_unscaled_enc
            errors[i, 0] = np.linalg.norm(sol_rot_unscaled_dec.reshape(-1,1) - sol_rot_unscaled.reshape(-1,1))**2
            errors_den[i, 0] = np.linalg.norm(u_test[:, i] + initial_state[:, 0])**2

            #if j == 12: 
            #    if i == 300:
            #        space = NumpyVectorSpace(2*Nx*Ny)
            #        experiment.fom.visualize(space.from_numpy(sol_rot_unscaled_dec.reshape(-1,1) + initial_state[:, 0].reshape(-1,1)))
            #        experiment.fom.visualize(space.from_numpy(u_test[:, i] + initial_state[:, 0]))
            #        experiment.fom.visualize(space.from_numpy(u_test[:, i] + initial_state[:, 0]) - space.from_numpy(sol_rot_unscaled_dec.reshape(-1,1) + initial_state[:, 0].reshape(-1,1)))


        proj_errors.append((j, np.sqrt(np.sum(errors, axis=0) / np.sum(errors_den, axis=0))[0]))
        

    if centered:
       # pod_results_file = filepaths['pod_results'] / f"proj_error_{Nx}x{Ny}_mu_{mu_val_tag}_nt_{config.nt}.pkl"
    
        # Target folder (relative or absolute)
        out_file = filepaths['pod_results'] / f"proj_error_pod_mu{mu_val_tag}.csv"

        with open(out_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y"])
            writer.writerows(proj_errors)
    else:

        pod_results_file = filepaths['pod_results'] / f"proj_error_uncentered_{Nx}x{Ny}_mu_{mu_val_tag}_nt_{config.nt}.pkl"

    #with pod_results_file.open("wb") as f:
    #    pickle.dump(proj_errors, f)
    
    print(proj_errors)

if __name__ == '__main__':
    proj_error_pod()    
