#!/usr/bin/env python

import time
import numpy as np
import pickle
import os
from pathlib import Path

from pymor.basic import *

import torch

from equiv_networks.autoencoders import GCNNAutoencoder2D, RotationGCNNAutoencoder2D, CNNAutoencoder2D, RotationUpsamplingGCNNAutoencoder2D
from equiv_networks.models.instationary.nonlinear_manifolds import NonlinearManifoldsMOR2D
from equiv_networks.models.instationary.deep_lspg_utilities_IMR import LSPG_quasi_newton
from scaling.scale import Scaler
from experiment_setup import WaveExperiment, WaveExperimentConfig


def test_wave_2D():
    """Test 2D wave equation with Deep-Galerkin method."""
    
    # Configure experiment
    config = WaveExperimentConfig(x_flow=True)
    experiment = WaveExperiment(config)
    scaled_data = True
    p_red = 12
    
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir = Path(script_dir)
    filepaths = experiment.get_filepath_patterns(script_dir)
    
    nn_save_filepath = filepaths['checkpoints'] / "wave_2D_CNN_p_12_256x256_t_16_11_2025-11_51_59_scaled_new_nolr_00005.pt"
    network_parameters_file = filepaths['network_parameters'] / "wave_2D_CNN_p_12_256x256_t_16_11_2025-11_51_59_scaled_new_nolr_00005.pkl"
    
    # Load network parameters
    with open(network_parameters_file, "rb") as f:
        parameters = pickle.load(f)
    
    # Verify filename consistency
    assert f"_{config.Nx}x{config.Ny}_" in str(nn_save_filepath)
    assert f"p_{p_red}_" in str(nn_save_filepath)
    
    # Setup model
    scaler = Scaler(dims=config.dims)
    model = NonlinearManifoldsMOR2D(network=CNNAutoencoder2D, scaler=scaler, dims=config.dims, network_parameters=parameters['network_parameters'])
    print("das sind die parameter:", parameters['network_parameters'])
    model.load_neural_network(path=nn_save_filepath)
    model.network.eval()
    
    trainable = sum(p.numel() for p in model.network.parameters() if p.requires_grad)
    print(f"Network has {trainable} trainable parameters")

    # Load test data
    # NOTE: Using precomputed snapshots for speed.
    # TODO take care that this actually matches
    mu_val = 1.5
    mu_test = experiment.fom.parameters.parse({'mu': mu_val})
    filename = filepaths['snapshots'] / "snapshots_256x256_sigpre_050_3_2_nt_1000_every_5_ts"
    with open(filename, 'rb') as f:
        arr = pickle.load(f)['snapshots']
    u_test = np.vstack(arr).T

    #NOTE: alternativ: compute u_test with x_flow False fom, but takes a lot longer
    if not config.x_flow:
        u_test = u_test.reshape(2, config.Nx, config.Ny, -1)
        u_test = np.rot90(u_test, k=-1, axes=(1,2)) #rotate countercockwise
        u_test = u_test.reshape(2*config.Nx*config.Ny, -1)

    # Compute initial condition and reference offset
    initial_state = experiment.get_initial_state(mu_val=mu_val)
    u_ref, u_0_hat = experiment.compute_reference_offset(model, mu_val=mu_val, scaled_data=scaled_data)
   
    u_approx = [u_0_hat]
    u_approx_full = [initial_state]

    #TODO this only needs to be done as we are loading data right now: 
    u_test = u_test + initial_state
    
    # Implicit midpoint timestepping for ROM
    print("Starting timestepping for ROM ...")
    for i in range(config.n_timesteps):
        tic = time.time()
        t = (i + 1) * config.dt
        print(f'Time: {t:.3f}')
        
        u_n1 = u_approx[-1]
        
        # Quasi-Newton solve
        u_new = LSPG_quasi_newton(model, u_n1, mu_test, config.dt, experiment.fom, u_ref, scaled_data, tol=1e-8)
        u_approx.append(u_new)
        
        # Decode to full-order solution
        decode_u_new = model.network.decode(torch.as_tensor(u_new, dtype=torch.double, device="cpu"))[0].detach().cpu().numpy()
        if scaled_data:
            decode_u_new = scaler.prolongate(scaler.unscale(decode_u_new))
        else:
            decode_u_new = scaler.prolongate(decode_u_new)

        if i == 13 or i==14:
            space = NumpyVectorSpace(model.dims[0]*model.dims[1]*model.dims[2])
            experiment.fom.visualize(space.make_array(u_ref + decode_u_new))
            experiment.fom.visualize(space.make_array(u_test[:, i]))
        
        u_approx_full.append((u_ref + decode_u_new).reshape(-1, 1))
        print(f"Step took {time.time() - tic:.2f}s")
    
    # Save results
    result_file = filepaths['mor_results'] / f'approx_full_deep_galerkin_p_{p_red}'
    with result_file.open('wb') as f:
        pickle.dump({'mu': mu_test, 'u_deep_galerkin': u_approx_full, 'u_full': u_test}, f)
    
    # Compute error metrics
    print("\n" + "="*60)
    print("Error Metrics:")
    print("="*60)
    
    metrics = experiment.compute_error_metrics(u_approx_full, u_test, u_approx_latent=u_approx, model=model)
    
    print(f"Relative error (total):  {metrics['relative_error_total']:.6e}")
    print(f"Relative error (q):      {metrics['relative_error_q']:.6e}")
    print(f"Relative error (p):      {metrics['relative_error_p']:.6e}")
    print(f"Relative error (latent): {metrics['relative_error_latent']:.6e}")
    
    # Save error to file
    error_file = filepaths['mor_results'] / "test_relative_errors_wave.txt"
    with error_file.open("a") as f:
        f.write(f"Deep-Galerkin\t{p_red}\t{mu_val}\t{metrics['relative_error_total']}\n")

if __name__ == '__main__':
    test_wave_2D()
