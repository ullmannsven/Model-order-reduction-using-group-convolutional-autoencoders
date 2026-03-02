#!/usr/bin/env python

import pickle
import os

from experiment_setup import WaveExperiment, WaveExperimentConfig

def create_snapshots(): 
    config = WaveExperimentConfig(Nx=256, Ny=5)
      
    #number_of_snapshots = 1
    #parameter_space = experiment.fom.parameters.space(0.5, 0.5)
    #train_params = parameter_space.sample_uniformly(number_of_snapshots)

    # we dont store every timestep, but only every {timestep_factor}th (in order to reduce trainingset size)
    # we use more timesteps to compute the FOM, as otherwise numerical instabilities occur
    timestep_factor = config.timestep_factor
    counter = 0

    data_tmp = []
    for mu_val in [0.6, 0.8]:
            experiment = WaveExperiment(config, mu_val=mu_val)
            print("was ist mu_val", mu_val)
            parameter_space = experiment.fom.parameters.space(mu_val, mu_val)
            mu = parameter_space.sample_uniformly(1)[0]
            print("was ist mu", mu)
            solution = experiment.fom.solve(mu)
            experiment.fom.visualize(solution)
            solution_mat = solution.to_numpy()
             # solution at first timestep
            solution_0 = solution_mat[:, 0]

            for j in range(config.n_timesteps):
                # this includes the zero vector in the training set, as is encouraged by Lee and Carlberg in Section 5.3
                data_tmp.append(solution_mat[:, timestep_factor*j] - solution_0)

            os.makedirs('snapshots_grid', exist_ok=True)
            filename = f'snapshots_grid/snapshots_{config.Nx}x{config.Ny}_{2}_{counter}_nt_{config.nt}_every_{timestep_factor}_ts'
            counter += 1

            with open(filename, 'wb') as file_obj:
               pickle.dump({'snapshots': data_tmp}, file_obj)
            data_tmp = [] 

if __name__ == '__main__':
    create_snapshots()
