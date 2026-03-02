import os 
from pathlib import Path
import pickle
import numpy as np

from pymor.basic import *

from experiment_setup import WaveExperimentConfig, WaveExperiment

config = WaveExperimentConfig(nt=500, timestep_factor=1, visualize_q=True)
experiment = WaveExperiment(config)

script_dir = os.path.dirname(os.path.abspath(__file__))
script_dir = Path(script_dir)
filepaths = experiment.get_filepath_patterns(script_dir)

mu_val = 0.8
#torch_path = filepaths['mor_results'] / "CNN_MG_p_12_xflow"
torch_path = filepaths['snapshots'] / "snapshots_256x256_mu0.6"

with open(torch_path, "rb") as file:
    snapshot = pickle.load(file)['snapshots']
    #snapshot = pickle.load(file)['u_deep_galerkin']


snapshot = np.vstack(snapshot).T
#snapshot = np.hstack(snapshot)

print(snapshot.shape)

q0, p0 = experiment._get_initial_condition()
initial_state = np.hstack((q0, - mu_val * p0)).reshape(-1, 1)
snapshot = snapshot + initial_state

space2 = NumpyVectorSpace(config.Nx * config.Ny * 2)

#experiment.fom.visualize(space2.from_numpy(snapshot))
experiment.fom.visualize(space2.from_numpy(snapshot[:, 0]))
experiment.fom.visualize(space2.from_numpy(snapshot[:, 100]))
experiment.fom.visualize(space2.from_numpy(snapshot[:, 500]))

