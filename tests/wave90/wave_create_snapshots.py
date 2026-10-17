#!/usr/bin/env python

import pickle
import os

from experiment_setup import WaveExperiment, WaveExperimentConfig

def create_snapshots(): 
    config = WaveExperimentConfig(x_flow=True, nt=500, timestep_factor=1)
    experiment = WaveExperiment(config)

    train_params = [0.5, 0.6, 0.75, 0.8, 1.0]
    data_tmp = []
    
    for i, mu_val in enumerate(train_params):
        mu = experiment.fom.parameters.parse(mu_val)
        solution = experiment.fom.solve(mu)
        print(f'{i}: {mu_val}')
        solution_mat = solution.to_numpy()
        # solution at first timestep
        solution_0 = solution_mat[:, 0]

        for j in range(config.nt):
            # this includes the zero vector in the training set, as is encouraged by Lee and Carlberg in Section 5.3
            data_tmp.append(solution_mat[:, j] - solution_0)

        os.makedirs('snapshots_grid', exist_ok=True)
        filename = f'snapshots_grid/snapshots_{config.Nx}x{config.Ny}_{mu_val}_nt_{config.nt}'
        with open(filename, 'wb') as file_obj:
            pickle.dump({'snapshots': data_tmp}, file_obj)
        data_tmp = []

if __name__ == '__main__':
    create_snapshots()