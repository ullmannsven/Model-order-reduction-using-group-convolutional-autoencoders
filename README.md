# Model order reduction using group convolutional autoencoders

This repository contains the numerical examples of my Master thesis **Model order reduction using group convolutional autoencoders** and implements group-equivariant convolutional autoencoders combined with structure-preserving model order reduction (MOR) techniques for parametric dynamical systems. The primary application is the 2D wave equation, where rotational symmetry (C4/C8 equivariance) is exploited to improve generalization across different flow directions.

The core idea is to train a neural network autoencoder on waves propagating in one direction and generalize to perpendicular directions without retraining.

---

## Project Structure

```
.
├── equiv_networks/                  # Core library: network architectures and MOR models
│   ├── autoencoders.py              # Autoencoder architecture definitions
|   ├── early_stopping.py            # Early stopping procedure via checkpointing
|   ├── trainer.py                   # Training prodedure of the autoencoders
│   ├── models/
│       ├── deep_galerkin_utilities_IMR.py  # Deep Galerkin quasi-Newton solver
│       ├── deep_lspg_utilities_IMR.py      # Deep LSPG quasi-Newton solver
|       ├── general_utilities.py            # Helpers that are employed in all files in the folder
|       └── nonlinear_manifolds.py          # MOR wrapper around the autoencoder
├── tests/                          
|    ├── 45wave/
|       ├── checkpoints/
|       ├── mor_results
|       ├── network_parameters/
|       ├── scaling/
|       ├── snapshots/grid/
|       ├── experiment_setup.py 
|       ├── proj_error_AE.py
|       ├── train_wave.py
        └── wave_create_snapshots.py
|    ├── 90wave/ 
|       ├── AE_results/
|       ├── checkpoints/
|       ├── CL_results/
|       ├── mor_results/
|       ├── network_parameters/
|       ├── pod_results/
|       ├── scaling/
|       ├── snapshots_grid/
|       ├── scaling/
|       ├── compute_cl_basis.py          # Compute Cotangent Lift reduced basis
|       ├── compute_pod_basis.py         # Compute POD reduced basis
|       ├── experiment_setup.py          # Experiment configuration and FOM setup
|       ├── proj_error_AE.py             # Projection error: autoencoder
|       ├── proj_error_cl.py             # Projection error: Cotangent Lift
|       ├── proj_error_pod.py            # Projection error: POD
│       ├── test_wave_cl_sg.py           # ROM test: CL + Galerkin projection
│       ├── test_wave_deep_galerkin.py   # ROM test: AE + Deep Galerkin
│       ├── test_wave_deep_lspg.py       # ROM test: AE + Deep LSPG
│       ├── test_wave_pod_galerkin.py    # ROM test: POD + Galerkin projection
│       ├── train_wave.py                # Autoencoder training
|       └── wave_create_snapshots        # Compute FOM solutions
```

---

## Core Library (`equiv_networks/`)

### `autoencoders.py`
Defines all autoencoder architectures used in the project. The key variants are:

| Name | Description |
|------|-------------|
| `RotationUpsamplingGCNNAutoencoder2D` | Group-equivariant autoencoder with C4 or C8 rotational symmetry, using ESCNN. The encoder uses group convolutions; the decoder uses upsampling transposed convolutions. |
| `UpsamplingCNNAutoencoder2D` | Standard (non-equivariant) CNN autoencoder baseline with the same upsampling decoder architecture. |
| `TrivialUpsamplingGCNNAutoencoder2D` | GCNN autoencoder with trivial (scalar) representations — equivariant in structure but without non-trivial group action on features. |
| `RotationUpsamplingGCNN2D_TorchOnly` | GCNN autoencoder for H=C_4, implemented using pyTorch and thus implementing group convolutions "by hand" instead of using escnn. |

All autoencoders share the same encode/decode interface and are selected via the `AE_REGISTRY` in the test scripts.

### `early_stopping.py` 
Early stopping procedure, that checkpoints the best trained model via computing the current validation loss. Includes a patience parameter P, i.e., if no improvment of the validation loss for P epochs occurs, training is terminated. 

### `trainer.py`
Contains the complete training routine for a neural network, in this case employed only for autoencoders. Different loss functions, including the MSE, a weighted MSE, a physical aware MSE and the symplectic loss term are included here. 

### `models/deep_galerkin_utilities_IMR`
Implements the **Deep Galerkin** time integration: a quasi-Newton solver (`Galerkin_quasi_newton`) for implicit midpoint rule (IMR) timestepping on the reduced nonlinear manifold. The residual is formulated via Galerkin projection of the FOM onto the reduced manifold via the trained autoencoder

### `models/deep_lspg_utilities_IMR.py`
Implements the **Deep LSPG** (Least-Squares Petrov-Galerkin) time integration: a quasi-Newton solver (`LSPG_quasi_newton`) for IMR timestepping. Differs from Galerkin in the test space used for projection — LSPG minimizes the full-order residual in a least-squares sense.

### `models/general_utlities.py`
Implements general helpers, including a `apply_decoder` (applies the decoder and the respective unscaling and prolongation operator) and a `get_jacobian`, which computes the jacobian of the AE-decoder via pyTorches `jacfwd` method. 

### `models/nonlinear_manifolds.py`
Wraps an autoencoder into a full MOR model (`NonlinearManifoldsMOR2D`). Handles loading/saving of network weights and interfacing with the FOM. 

---

## Experiment for the pure right moving wave (`tests/90wave/`)

### `tests/90wave/AE_results/` 

Projection results when using autoencoders. Contains results regarding GCNN, UpsamplingCNN and UpsamplingCNN_symplectic for the test parameters `mu ∈ {0.6, 0.8}`. The mean of these two is stored in seperate files (named without the mu-value at the end).

### `tests/90wave/checkpoints/` 

Checkpoints of the selected autoencoders for different AE setups and reduced basis sizes. 

### `tests/90wave/CL_results/` 

Results when using CL as projection method as well as CL-SG as reduction method. Contains projections error, reduction errors as well as the computed RB using CL (once centered and once uncentered). 

### `tests/90wave/mor_results/` 

Results when using different autoencoder variants. 

### `tests/90wave/network_parameters/`
Contains the network parameters that corresponds to the checkpoints in `tests/90wave/checkpoints`. 

### `tests/90wave/pod_results/` 

Results regarding experiments when using POD as projection method as well as POD-G as reduction method. Contains projections error, reduction errors as well as the computed RB using CL (once centered and once uncentered).

### `tests/90wave/scaling/` 
Contains `scale.py`, which performs scaling (and inverse scaling) and the reshaping (and prolongation).

### `tests/90wave/snapshots_grid/`

Currently empty as the FOM solutions are too large to store on github. Then FOM solutions can be obtained using `wave_create_snapshots`. 


## Basis Computation

These scripts precompute reduced bases from training snapshots. We ran them once before any ROM tests.

### `compute_cl_basis.py` — Cotangent Lift Basis
Loads training snapshots for `mu ∈ {0.5, 0.75, 1.0}`, assembles a phase-space snapshot matrix, and computes a symplectic reduced basis using pyMOR's `psd_cotangent_lift`. Saves the basis as a pickle file.

```bash
python compute_cl_basis.py --max_modes 50 --centered
```

### `compute_pod_basis.py` — POD Basis
Loads the same training snapshots and computes a standard POD basis using pyMOR's `pod` function. Saves the basis as a `.npy` file.

```bash
python compute_pod_basis.py --modes 50 --centered
```

---

## Experiment setup

### `tests/90wave/experiment_setup.py` 

Central configuration file for all experiments regarding the right moving wave. Contains:

- **`WaveExperimentConfig`** — dataclass holding all experiment hyperparameters: grid size (`Nx`, `Ny`), number of timesteps (`nt`), timestep factor, flow direction (`x_flow`), and visualization flags.
- **`WaveExperiment`** — sets up the pyMOR full-order model (FOM) for the 2D wave equation, provides helper methods for loading initial conditions, computing reference offsets, and evaluating error metrics.
- **`get_filepath_patterns`** — returns a dict of standardized paths for snapshots, checkpoints, network parameters, and results directories.

---


## Projection Error Evaluation

These scripts evaluate how well each reduced representation can approximate test snapshots, without running a time integrator. They measure the offline approximation quality of each method.

### `proj_error_AE.py` — Autoencoder Projection Error
Encodes and decodes test snapshots through the trained autoencoder and computes the relative reconstruction error for each latent dimension size.

```bash
python proj_error_AE.py --ae_name RotationUpsamplingGCNN_C8 --p_red 4 8 12 16 --mu_val 0.8 --write_csv False
```

### `proj_error_cl.py` — Cotangent Lift Projection Error
Projects test snapshots onto the CL reduced basis using symplectic projection and computes the relative error for each basis size.

```bash
python proj_error_cl.py --p_red 4 8 12 16 --mu_val 0.8 --rb_size 50 --write_csv False
```

### `proj_error_pod.py` — POD Projection Error
Projects test snapshots onto the POD basis via standard orthogonal projection and computes the relative error for each basis size.

```bash
python proj_error_pod.py --p_red 4 8 12 16 --mu_val 0.6 --rb_size 50 --centered --write_csv
```

---

## ROM Time Integration Tests

These scripts run a full reduced-order model time integration and compare against reference snapshots.

### `test_wave_cl_sg.py` — Cotangent Lift + symplectic Galerkin (ueses pyMOR reductor)
Uses pyMOR's `QuadraticHamiltonianRBReductor` to build and solve a symplectic Galerkin ROM on the CL basis. Computes reconstruction errors against test snapshots.

```bash
python test_wave_cl_sg.py --p_red 4 8 12 16 --mu_val 0.6 --rb_size 50 --visualize --save_data
```

### `test_wave_deep_galerkin.py` — Autoencoder + Deep Galerkin
Runs the Deep Galerkin ROM: the autoencoder defines the nonlinear reduced manifold, and the implicit midpoint rule is solved via quasi-Newton iteration with Galerkin projection.

```bash
python test_wave_deep_galerkin.py --ae_name RotationUpsamplingGCNN_C8 --p_red 8 --mu_val 0.8 --visualize
```

### `test_wave_deep_lspg.py` — Autoencoder + Deep LSPG
Same as Deep Galerkin but uses LSPG projection (minimizes the full-order residual) instead of Galerkin projection. Generally more robust but more expensive per timestep.

```bash
python test_wave_deep_lspg.py --ae_name RotationUpsamplingGCNN_C8 --p_red 8 --mu_val 0.8 --visualize --save_data
```

### `test_wave_pod_galerkin.py` — POD + Galerkin (uses pyMOR reductor)
Uses pyMOR's `InstationaryRBReductor` to build and solve a standard Galerkin ROM on the POD basis.

```bash
python test_wave_pod_galerkin.py --p_red 4 8 12 16 --mu_val 0.8 --rb_size 50 --visualize --save_data
```

## Training 

Two files are required to perform training: 

### `train_wave.py` 
Sets all network hyperparameters and trains the autoencoders.

### `wave_create_snapshots` 
Before training, training data needs to be generated. This script generates the training (and also the test) data. Run this once before starting training. 

---

## Experiment for diagonal moving wave (`tests/45wave/`)

Structured in a similar fashion as the `tests/90wave`, therefore no additional detailed description is provided here. 


--- 

## Autoencoder Registry

All test scripts that use a trained autoencoder share the same `AE_REGISTRY`, which maps a short name to the corresponding network class and group space:

| `--ae_name` | Network class | Group |
|-------------|---------------|-------|
| `RotationUpsamplingGCNN_C4` | `RotationUpsamplingGCNNAutoencoder2D` | C4 (4 rotations) |
| `RotationUpsamplingGCNN_C8` | `RotationUpsamplingGCNNAutoencoder2D` | C8 (8 rotations) |
| `UpsamplingCNN` | `UpsamplingCNNAutoencoder2D` | None (baseline) |
| `TrivialUpsamplingGCNN` | `TrivialUpsamplingGCNNAutoencoder2D` | Trivial |

Checkpoint files are expected to follow the naming convention:
```
checkpoints/wave_2D_{ae_name}_p_{p_red}_{Nx}x{Ny}.pt
network_parameters/wave_2D_{ae_name}_p_{p_red}_{Nx}x{Ny}.pkl
```

---

## Dependencies

- [pyMOR](https://pymor.org/) — model order reduction framework (FOM, reducers, symplectic methods)
- [ESCNN](https://github.com/QUVA-Lab/escnn) — equivariant steerable CNNs (group convolutions)
- [PyTorch](https://pytorch.org/) — neural network training and inference
- NumPy, SciPy — numerical computations
