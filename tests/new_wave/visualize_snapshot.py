import os 
from pathlib import Path
import pickle
import numpy as np

from pymor.basic import *

from experiment_setup import WaveExperimentConfig, WaveExperiment

mu_val = 1

config = WaveExperimentConfig(x_flow=True, nt=500, timestep_factor=1, visualize_q=False)
experiment = WaveExperiment(config, mu_val=mu_val)

script_dir = os.path.dirname(os.path.abspath(__file__))
script_dir = Path(script_dir)


base_dir = os.path.join(script_dir, "snapshots_grid")
base_dir = Path(base_dir)
os.makedirs(base_dir, exist_ok=True)

with open(base_dir / f"snapshots_256x256_100_nt_500", "rb") as file:
    snapshot = pickle.load(file)['snapshots']

snapshot = np.vstack(snapshot).T

q0, p0 = experiment._get_initial_condition(mu_val=mu_val)
initial_state = np.hstack((q0, - mu_val * p0)).reshape(-1, 1)

snapshot = snapshot + initial_state

space2 = NumpyVectorSpace(config.Nx * config.Ny * 2)

experiment.fom.visualize(space2.from_numpy(snapshot))

