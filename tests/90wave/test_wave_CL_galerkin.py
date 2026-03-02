#!/usr/bin/env python

import time
import numpy as np
import pickle
import os
from pathlib import Path

from pymor.basic import *

from equiv_networks.models.instationary.pod_galerkin_utilities_IMR import POD_Galerkin_quasi_newton
from experiment_setup import WaveExperimentConfig, WaveExperiment

def test_wave_2D(): 

    # Configure experiment
    config = WaveExperimentConfig()
    experiment = WaveExperiment(config)
    p_red = 20

    script_dir = os.path.dirname(os.path.abspath(__file__))
    rb_dir = os.path.join(script_dir, "CL_results")
    script_dir = Path(script_dir)
    rb_dir = Path(rb_dir)
    os.makedirs(rb_dir, exist_ok=True)

    filepaths = experiment.get_filepath_patterns(script_dir)
    
    # Load test data
    # NOTE: Using precomputed snapshots for speed.
    #TODO take care that this actually matches
    mu_test_val = 1.2
    mu_test = experiment.fom.parameters.parse({'mu': mu_test_val})
    filename = filepaths['snapshots'] / "snapshots_256x256_sigpre_050_5_3_nt_1000_every_5_ts"
    with open(filename, 'rb') as f:
        arr = pickle.load(f)['snapshots']
    u_test = np.vstack(arr).T

    rb_path = rb_dir / f"reduced_basis_CL_centered_256x256_sigpre_050_rbsize_50_nt_1000_every_5_ts"
    with open(rb_path, "rb") as file:
        reduced_basis = pickle.load(file)

    rb = reduced_basis[:p_red//2]
    rb_array = rb.to_array().to_numpy()
    rb_tsi = rb.transposed_symplectic_inverse().to_array()

    initial_state = experiment.get_initial_state(mu_val=mu_test_val)
    u_ref = initial_state

    u_approx = [np.zeros((1, p_red))]
    u_approx_full = [initial_state]

    #TODO this only needs to be done as we are loading data right now: 
    u_test = u_test + initial_state

    # Implicit midpoint timestepping for ROM
    for i in range(config.n_timesteps):
        tic = time.time()
        t = (i + 1) * config.dt
        print(f'Time: {t}')
        u_n1 = u_approx[-1]

        ## Quasi Newton:
        u_new = POD_Galerkin_quasi_newton(rb_array, u_n1, mu_test, config.dt, experiment.fom, u_ref, tol=1e-8)
        ## End Quasi Newton

        u_approx.append(u_new)
        decode_u_new = rb_array @ u_new.T
        u_approx_full.append((u_ref + decode_u_new).reshape(-1,1))
        print(f"Step took {time.time()-tic}")

    # Compute error metrics
    print("\n" + "="*60)
    print("Error Metrics:")
    print("="*60)
    
    metrics = experiment.compute_error_metrics(u_approx_full, u_test)
    
    print(f"Relative error (total):  {metrics['relative_error_total']:.6e}")
    print(f"Relative error (q):      {metrics['relative_error_q']:.6e}")
    print(f"Relative error (p):      {metrics['relative_error_p']:.6e}")
    
    # Save error to file
    error_file = filepaths['mor_results'] / "test_relative_errors_wave.txt"
    with error_file.open("a") as f:
        f.write(f"POD_LSPG\t{p_red}\t{mu_test_val}\t{metrics['relative_error_total']}\n")


if __name__ == '__main__':
    test_wave_2D()
