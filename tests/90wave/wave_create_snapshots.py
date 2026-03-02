#!/usr/bin/env python

import pickle
import os
import time 

from experiment_setup import WaveExperiment, WaveExperimentConfig

def create_snapshots(): 
    config = WaveExperimentConfig(x_flow=True, nt=500, timestep_factor=1)
    experiment = WaveExperiment(config)
    train_params = [2]

    data_tmp = []
    
    for i, mu_val in enumerate(train_params):
        mu = experiment.fom.parameters.parse(mu_val)
        solution = experiment.fom.solve(mu)
        print(f'{i}: {mu_val}')
        solution_mat = solution.to_numpy()
            # solution at first timestep
        solution_0 = solution_mat[:, 0]

        for j in range(config.nt):
            data_tmp.append(solution_mat[:, j] - solution_0)

        os.makedirs('snapshots_grid', exist_ok=True)
        
        filename = f'snapshots_grid/snapshots_{config.Nx}x{config.Ny}_{mu_val}_nt_{config.nt}'
        with open(filename, 'wb') as file_obj:
            pickle.dump({'snapshots': data_tmp}, file_obj)
        data_tmp = []

    # as inituition, visualize the solution for the last mu
    #experiment.fom.visualize(solution)

if __name__ == '__main__':
    tic = time.time()
    create_snapshots()
    print(time.time() - tic)
