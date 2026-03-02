#!/usr/bin/env python

import time
import numpy as np
import pickle
import os
from pathlib import Path

from pymor.basic import *
from pymor.reductors.symplectic import QuadraticHamiltonianRBReductor
from pymor.algorithms.timestepping import ImplicitMidpointTimeStepper
from experiment_setup import WaveExperimentConfig, WaveExperiment

def test_wave_cl_galerkin(): 

    # Configure experiment
    config = WaveExperimentConfig(nt=500, timestep_factor=1, visualize_q=True)
    experiment = WaveExperiment(config, mu_val=100) #dummy
    p_red = 20

    Nx = config.Nx
    Ny = config.Ny

    script_dir = os.path.dirname(os.path.abspath(__file__))
    rb_dir = os.path.join(script_dir, "CL_results")
    script_dir = Path(script_dir)
    rb_dir = Path(rb_dir)
    os.makedirs(rb_dir, exist_ok=True)

    filepaths = experiment.get_filepath_patterns(script_dir)

    # Load test data
    # NOTE: Using precomputed snapshots for speed.
    # TODO take care that this actually matches
    mu_test_val = 0.8
    experiment = WaveExperiment(config, mu_val=mu_test_val)
    mu_test = experiment.fom.parameters.parse({'mu': mu_test_val})
    filename = filepaths['snapshots'] / f"snapshots_{Nx}x{Ny}_080_nt_{config.nt}"
    with open(filename, 'rb') as f:
        arr = pickle.load(f)['snapshots']
    u_test = np.vstack(arr).T

    q0, p0 = experiment._get_initial_condition(mu_val=mu_test_val)
    initial_state = np.hstack((q0, -mu_test_val * p0)).reshape(-1, 1)
    u_test = u_test + initial_state

    rb_path = rb_dir / f"reduced_basis_uncentered_CL_{Nx}x{Ny}_rbsize_50_nt_{config.nt}"
    with open(rb_path, 'rb') as file:
        reduced_basis = pickle.load(file)

    #TODO understand why this line here is required
    RB = reduced_basis[:p_red//2]
   
    
    reductor = QuadraticHamiltonianRBReductor(experiment.fom.with_(nt=config.n_timesteps, time_stepper=ImplicitMidpointTimeStepper(nt=config.n_timesteps)), RB)
    rom = reductor.reduce()
    u_rom = rom.solve(mu=mu_test)

    space2 = NumpyVectorSpace(2*Nx*Ny)   
    experiment.fom.visualize(reductor.reconstruct(u_rom))
    experiment.fom.visualize(space2.make_array(u_test))

    error = np.sqrt(np.sum(np.linalg.norm(u_test - reductor.reconstruct(u_rom).to_numpy().reshape(2*Nx*Ny, -1)[:, :-1], axis=0)**2))
    error_den = np.sqrt(np.sum(np.linalg.norm(u_test)**2))

    error_q = np.sqrt(np.sum(np.linalg.norm(u_test[:Nx*Ny, :] - reductor.reconstruct(u_rom).to_numpy().reshape(2*Nx*Ny, -1)[:Nx*Ny, :-1], axis=0)**2))
    error_den_q = np.sqrt(np.sum(np.linalg.norm(u_test[:Nx*Ny, :])**2))

    error_p = np.sqrt(np.sum(np.linalg.norm(u_test[Nx*Ny:, :] - reductor.reconstruct(u_rom).to_numpy().reshape(2*Nx*Ny, -1)[Nx*Ny:, :-1], axis=0)**2))
    error_den_p = np.sqrt(np.sum(np.linalg.norm(u_test[Nx*Ny:, :])**2))

    print("was ist der Fehler", error/error_den)
    print("was ist der Fehler", error_q/error_den_q)
    print("was ist der Fehler", error_p/error_den_p)
    
if __name__ == '__main__':
    test_wave_cl_galerkin()
