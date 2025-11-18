#!/usr/bin/env python

import pickle
import os

from torchpdes.pdes.instationary import wave_2D

def create_snapshots():
    T = 1
    Nx = 256
    Ny = 5
    nt = 1000
    sig_pre = 0.5
    sig_pre_tag = f"{sig_pre:.2f}".replace('.', '')
    fom = wave_2D(T=T, Nx=Nx, Ny=Ny, nt=nt, sig_pre=sig_pre, x_flow=True)

    number_of_snapshots = 1
    parameter_space = fom.parameters.space(1.4, 1.4)
    train_params = parameter_space.sample_uniformly(number_of_snapshots)

    # we dont store every timestep, but only every {timestep_factor}th (in order to reduce trainingset size)
    # we use more timesteps to compute the FOM, as otherwise numerical instabilities occur
    timestep_factor = 5

    data_tmp = []
    for i, mu in enumerate(train_params):
            solution = fom.solve(mu)
            print(f'{i}: {mu}')
            solution_mat = solution.to_numpy()
             # solution at first timestep
            solution_0 = solution_mat[:, 0]

            for j in range(int(solution_mat.shape[1]/timestep_factor)):
                # this includes the zero vector in the training set, as is encouraged by Lee and Carlberg in Section 5.3
                data_tmp.append(solution_mat[:, timestep_factor*j] - solution_0)

            #os.makedirs('snapshots_grid', exist_ok=True)
            # TODO: take care, cant pass variable name of sig_pre, as it contain a decimal. Somehow this is weird behavior. 
            #filename = f'snapshots_grid/snapshots_{Nx}x{Ny}_sigpre_{sig_pre_tag}_{number_of_snapshots}_{i}_nt_{nt}_every_{timestep_factor}_ts_rename_tomorrow'

            #with open(filename, 'wb') as file_obj:
            #   pickle.dump({'snapshots': data_tmp}, file_obj)
            #data_tmp = []

    # as inituition, visualize the solution for the last mu
    fom.visualize(solution)

if __name__ == '__main__':
    create_snapshots()
