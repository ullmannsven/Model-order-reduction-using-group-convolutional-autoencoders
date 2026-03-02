#!/usr/bin/env python

import time
import numpy as np
import pickle
import os
from pathlib import Path

from pymor.basic import *

from equiv_networks.models.instationary.pod_lspg_utilities_IMR import POD_LSPG_quasi_newton
from experiment_setup import WaveExperimentConfig, WaveExperiment


def test_wave_2D():

    # Configure experiment
    config = WaveExperimentConfig(visualize_q=False)
    experiment = WaveExperiment(config)
    p_red = 20

    script_dir = os.path.dirname(os.path.abspath(__file__))
    rb_dir = os.path.join(script_dir, "pod_results")
    script_dir = Path(script_dir)
    rb_dir = Path(rb_dir)
    os.makedirs(rb_dir, exist_ok=True)

    filepaths = experiment.get_filepath_patterns(script_dir)
    
    # Load test data
    # NOTE: Using precomputed snapshots for speed.
    #TODO take care that this actually matches
    mu_test_val = 1.4
    mu_test = experiment.fom.parameters.parse({'mu': mu_test_val})
    filename = filepaths['snapshots'] / "snapshots_256x256_sigpre_050_5_4_nt_1000_every_5_ts"
    with open(filename, 'rb') as f:
        arr = pickle.load(f)['snapshots']
    u_test = np.vstack(arr).T


    rb_path = rb_dir / f"reduced_basis_{config.Nx}x{config.Ny}_sigpre_{config.sig_pre}_rbsize_20_nt_{config.nt}_every_{config.timestep_factor}_ts.npy"
    reduced_basis_all = np.load(rb_path)
    reduced_basis = reduced_basis_all[:, :p_red]

    #This is not doing anything, just to keep it aligned with the AE version
    zero_vec = np.zeros(2*config.Nx*config.Ny)
    u_0_hat = reduced_basis.T @ zero_vec
    decoded_u_0_hat = reduced_basis @ u_0_hat

    initial_state = experiment.get_initial_state(mu_val=mu_test_val)
    u_ref = initial_state - decoded_u_0_hat.reshape(-1,1)

    u_approx = [u_0_hat.reshape(1,-1)]
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
        u_new = POD_LSPG_quasi_newton(reduced_basis, u_n1, mu_test, config.dt, experiment.fom, u_ref, tol=1e-8)
        ## End Quasi Newton

        u_approx.append(u_new)
        decode_u_new = reduced_basis @ u_new.T
        u_approx_full.append((u_ref + decode_u_new).reshape(-1,1))
        print(f"Step took {time.time()-tic}")

        if i == 50 or i == 150:
            space = NumpyVectorSpace(reduced_basis.shape[0])
            experiment.fom.visualize(space.make_array(u_ref + decode_u_new))
            experiment.fom.visualize(space.make_array(u_test[:, i]))

    # Save results
    result_file = filepaths['mor_results'] / f'approx_full_pod_lspg_p_{p_red}'
    with result_file.open('wb') as f:
        pickle.dump({'mu': mu_test, 'u_pod_lspg': u_approx_full, 'u_full': u_test}, f)
    
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
