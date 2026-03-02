#!/usr/bin/env python

"""Centralized experiment setup for 2D wave equation experiments with outflow boundaries."""

import numpy as np
import torch
from dataclasses import dataclass

from pymor.basic import *

from pymor.operators.constructions import VectorOperator
from pymor.parameters.base import Parameters
from pymor.models.symplectic import QuadraticHamiltonianModel
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.discretizers.builtin.gui.visualizers import PatchVisualizer

import numpy as np
import scipy.sparse as sp

from pymor.models.symplectic import QuadraticHamiltonianModel
from pymor.vectorarrays.numpy import NumpyVectorSpace

from pymor.operators.numpy import NumpyMatrixOperator
from pymor.operators.constructions import LincombOperator
from pymor.operators.block import BlockDiagonalOperator, BlockColumnOperator
from pymor.parameters.functionals import ExpressionParameterFunctional


@dataclass
class WaveExperimentConfig:
    """Configuration for wave equation experiments."""
    # Grid parameters
    Nx: int = 256
    Ny: int = 256
    Lx: float = 1.0
    Ly: float = 1.0
    
    # Time parameters
    T: float = 1.0
    nt: int = 1000
    timestep_factor: int = 5
     
    # Initial condition parameters
    x0: float = 0.3
    y0: float = 0.3
    sig: float = 0.05

    # visualization flag
    visualize_q: bool = True
    x_flow: bool = True
      
    @property
    def hx(self):
        """Grid spacing in x-direction."""
        if self.x_flow:
            return self.Lx / self.Nx
        else:
            return self.Lx / (self.Nx - 1)
    
    @property
    def hy(self):
        """Grid spacing in y-direction."""
        if self.x_flow:
            return self.Ly / (self.Ny - 1)
        else:
            return self.Ly / self.Ny
    
    @property
    def dims(self):
        """Dimensions tuple (channels, Nx, Ny)."""
        return (2, self.Nx, self.Ny)
    
    @property
    def dt(self):
        """Time step size."""
        return self.timestep_factor / self.nt
    
    @property
    def n_timesteps(self):
        """Number of reduced timesteps."""
        return int(self.T * self.nt / self.timestep_factor)


class WaveExperiment:
    """Centralized experiment setup for 2D wave equation experiments."""
    
    def __init__(self, config, mu_val):
        """Initialize wave experiment.
        
        Args:
            config: Experiment configuration. If None, uses default values.
        """
        self.config = config
        self.mu_val = mu_val
        self._fom = None
    
    @property
    def fom(self):
        """Create the full order model."""
        if self._fom is None:
            fom = self.build_wave_2D_fom(self.mu_val)
            self._fom = fom
        return self._fom
        
    
    def get_grid(self):
        """Get the spatial grid.
        
        Returns:
            Tuple of (x, y) coordinate arrays.
        """
        if self.config.x_flow:
            x = np.linspace(0, self.config.Lx, self.config.Nx, endpoint=False)
            y = np.linspace(0, self.config.Ly, self.config.Ny, endpoint=True)
        else:
            x = np.linspace(0, self.config.Lx, self.config.Nx, endpoint=True)
            y = np.linspace(0, self.config.Ly, self.config.Ny, endpoint=False)
        return x, y

    def compute_derivative(self, Q, direction):
        """Compute spatial derivative with appropriate boundary conditions.
        
        Parameters:
            Q: 
                Array of shape (Ny, Nx) containing field values.
            direction: 
                'x_flow' or 'y_flow' for derivative direction.
            
        Returns:
            Derivative array of same shape as Q.
        """
        dQ = np.zeros_like(Q)
        
        if direction == 'x_flow':
            h = self.config.hx
            # Central differences with periodic BC in x
            dQ[:, 1:-1] = (Q[:, 2:] - Q[:, :-2]) / (2 * h)
            dQ[:, 0] = (Q[:, 1] - Q[:, -1]) / (2 * h)
            dQ[:, -1] = (Q[:, 0] - Q[:, -2]) / (2 * h)

        elif direction == 'y_flow':
            h = self.config.hy
            # Central differences with periodic BC in y
            dQ[1:-1, :] = (Q[2:, :] - Q[:-2, :]) / (2 * h)
            dQ[0, :] = (Q[1, :] - Q[-1, :]) / (2 * h)
            dQ[-1, :] = (Q[0, :] - Q[-2, :]) / (2 * h)
        else:
            raise ValueError(f"Invalid direction: {direction}")
        
        return dQ
    

    def _get_initial_condition(self, mu_val):
        """Compute initial condition (q0, p0).
        
        Returns:
            Tuple of (q0_flat, p0_flat) as 1D arrays.
        """ 
        x, y = self.get_grid()
        X, Y = np.meshgrid(x, y, indexing='xy')
        
        if self.config.x_flow:
            # Gaussian pulse in x-direction
            q0_flat = np.exp(-((X - self.config.x0)**2) / (2 * mu_val**2 * self.config.sig**2)).ravel()
            
            Q = q0_flat.reshape(self.config.Ny, self.config.Nx)
            dqdx_mat = self.compute_derivative(Q, direction='x_flow')
            p0_flat = dqdx_mat.ravel()
        else:
            # Gaussian pulse in y-direction
            q0_flat = np.exp(-((Y - self.config.y0)**2) / (2 * mu_val**2 * self.config.sig**2)).ravel()
            
            Q = q0_flat.reshape(self.config.Ny, self.config.Nx)
            dqdy_mat = self.compute_derivative(Q, direction='y_flow')
            p0_flat = dqdy_mat.ravel()
        
        return (q0_flat, p0_flat)
     
    def compute_reference_offset(self, model, mu_val, scaled_data):
        """Compute reference offset u_ref = u_0 - decoder(encoder(0)).
        
        Args:
            model: Model with network that has encode/decode methods.
            zero_state: Optional zero state. If None, uses zeros of correct size.
            
        Returns:
            Reference offset as array of shape (2*Nx*Ny, 1).
        """

        zero_state = np.zeros(2 * self.config.Nx * self.config.Ny)
        
        # Encode and decode zero state
        if scaled_data:
            zero_tensor = torch.as_tensor(model.scaler.scale(model.scaler.restrict(zero_state)), dtype=torch.double, device="cpu").unsqueeze(0)
        else:
            zero_tensor = torch.as_tensor(model.scaler.restrict(zero_state), dtype=torch.double, device="cpu").unsqueeze(0)
        
        u_0_hat = model.network.encode(zero_tensor).detach().cpu().numpy()
        decoded_u_0_hat = model.network.decode(torch.as_tensor(u_0_hat, dtype=torch.double, device="cpu"))[0, :, :, :].detach().cpu().numpy()
        if scaled_data:
            decoded_u_0_hat = model.scaler.prolongate(model.scaler.unscale(decoded_u_0_hat)).detach().cpu().numpy()
        else:
            decoded_u_0_hat = model.scaler.prolongate(decoded_u_0_hat).detach().cpu().numpy()
        
        # Compute reference offset
        q0, p0 = self._get_initial_condition(mu_val=mu_val)
        initial_state = np.hstack((q0, - mu_val * p0)).reshape(-1, 1)
        u_ref = initial_state - decoded_u_0_hat
        
        return u_ref, initial_state, u_0_hat
    
    def build_wave_2D_fom(self, mu_val):
        """ Discretized linear wave equation in 2D posed on [0,1]^2 as a Hamiltonian system 
        with outflow boundary conditions on all sides.
        
        The boundary conditions allow waves to exit the domain without reflection.
        This is implemented using one-sided derivative stencils at boundaries.
        """
        T = self.config.T
        Nx = self.config.Nx
        Ny = self.config.Ny
        nt = self.config.nt
        visualize_q = self.config.visualize_q
    
        hx = self.config.hx
        hy = self.config.hy
        mu = ExpressionParameterFunctional('mu', Parameters({'mu': 1}), name='mu')

        def D2_neumann_1d(n, h):
            # interior: (1,-2,1)/h^2
            # boundary rows: (-1,1)/h^2 implementing mirrored ghost (u'_n=0)
            main = -2*np.ones(n)
            off = 1*np.ones(n-1)
            A = sp.diags([off, main, off], offsets=[-1,0,1], format='csr')
            A[0,0] = -1
            A[0,1] = 1
            A[-1,-1] = -1
            A[-1,-2] = 1
            
            return A.tocsr() / (h**2)
        
        def D2_periodic_1d(n, h):
            main = -2*np.ones(n)
            off = 1*np.ones(n-1)
            A = sp.diags([off, main, off], offsets=[-1,0,1], format='csr')
            A[0, -1] = 1
            A[-1, 0] = 1
            
            return A.tocsr() / (h**2)
        
        Ix = sp.eye(Nx, format='csr')
        Iy = sp.eye(Ny, format='csr')

        # Laplacian: Δ = d^2/dx^2 + d^2/dy^2
        if self.config.x_flow:
            Dxx = D2_periodic_1d(Nx, hx)
            Dyy = D2_neumann_1d(Ny, hy)
        else: 
            Dxx = D2_neumann_1d(Nx, hx)
            Dyy = D2_periodic_1d(Ny, hy)

        Lapl = (sp.kron(Iy, Dxx) + sp.kron(Dyy, Ix)).tocsr()
    
        # Hamiltonian operator H_op = diag(- c^2 K, I) on the block space
        Hqq = LincombOperator(operators=[NumpyMatrixOperator(Lapl)], coefficients=[(-1) * mu * mu])
        Hpp = NumpyMatrixOperator(sp.eye(Nx*Ny, format='csr'))
        H_op = BlockDiagonalOperator([Hqq, Hpp])

        V = NumpyVectorSpace(Nx*Ny)
        q0, p0 = self._get_initial_condition(mu_val)
        u0 = BlockColumnOperator([VectorOperator(V.from_numpy(q0)), LincombOperator([VectorOperator(V.from_numpy(p0))], [(-1) * mu])])

        class QBlockVisualizer:
            """Adapter that visualizes only the q-block of a (q,p) state on a RectGrid."""
            def __init__(self, Nx, Ny):
                self.Nx = Nx
                self.Ny = Ny
                self.grid = RectGrid(domain=([0,0], [1,1]), num_intervals=(Nx-1, Ny-1))
                self.base = PatchVisualizer(grid=self.grid, codim=2)
                self.scalar_space = NumpyVectorSpace(Nx*Ny)

            def visualize(self, U, **kwargs):
                data = self.scalar_space.empty()
                for i in range(len(U)):
                    ui = U[i].to_numpy()
                    q_flat = ui[:self.Nx*self.Ny]
                    data.append(self.scalar_space.from_numpy(q_flat))

                return self.base.visualize(data, **kwargs)
            
        class PBlockVisualizer:
            """Adapter that visualizes only the p-block of a (q,p) state on a RectGrid."""
            def __init__(self, Nx, Ny):
                self.Nx = Nx
                self.Ny = Ny
                self.grid = RectGrid(domain=([0,0], [1,1]), num_intervals=(Nx-1, Ny-1))
                self.base = PatchVisualizer(grid=self.grid, codim=2)
                self.scalar_space = NumpyVectorSpace(Nx*Ny)

            def visualize(self, U, **kwargs):
                data = self.scalar_space.empty()
                for i in range(len(U)):
                    ui = U[i].to_numpy()
                    p_flat = ui[self.Nx*self.Ny:]
                    data.append(self.scalar_space.from_numpy(p_flat))

                return self.base.visualize(data, **kwargs)
            
        # build the Hamiltonian model
        nt = int(T * nt)

        if visualize_q:
            model = QuadraticHamiltonianModel(T=T, initial_data=u0, H_op=H_op, visualizer=QBlockVisualizer(Nx=Nx, Ny=Ny), nt=nt)
        else: 
            model = QuadraticHamiltonianModel(T=T, initial_data=u0, H_op=H_op, visualizer=PBlockVisualizer(Nx=Nx, Ny=Ny), nt=nt)

        return model
    
    def get_filepath_patterns(self, base_dir):
        """Get standard filepath patterns for this experiment.
        
        Args:
            base_dir: Base directory for the experiment.
        Returns:
            Dictionary with keys: 'checkpoints', 'network_parameters',
            'snapshots', 'mor_results'.
        """
        patterns = {
            'checkpoints': base_dir / "checkpoints",
            'network_parameters': base_dir / "network_parameters",
            'snapshots': base_dir / "snapshots_grid",
            'mor_results': base_dir / "mor_results"
        }
        return patterns
    

    def compute_error_metrics(self, u_approx_full, u_test):
        """Compute error metrics between approximate and exact solutions.
        
        Args:
            u_approx_full: List of approximate full-order solutions.
            u_test: Exact solution array of shape (2*Nx*Ny, n_timesteps).
            u_approx_latent: Optional list of latent approximations.
            model: Optional model for computing latent errors.
            
        Returns:
            Dictionary with error metrics.
        """
        Nx, Ny = self.config.Nx, self.config.Ny
        n_steps = self.config.n_timesteps
        
        error_sum = 0.0
        error_sum_q = 0.0
        error_sum_p = 0.0
        norm_sum = 0.0
        norm_sum_q = 0.0
        norm_sum_p = 0.0
        
        for i in range(n_steps):
            # Full-order errors
            error_sum += np.linalg.norm(u_approx_full[i] - u_test[:, i].reshape(-1, 1))**2
            norm_sum += np.linalg.norm(u_test[:, i])**2
            
            # q-component errors
            error_sum_q += np.linalg.norm(u_approx_full[i][:Nx*Ny] - u_test[:Nx*Ny, i].reshape(-1, 1))**2
            norm_sum_q += np.linalg.norm(u_test[:Nx*Ny, i])**2
            
            # p-component errors
            error_sum_p += np.linalg.norm(u_approx_full[i][Nx*Ny:] - u_test[Nx*Ny:, i].reshape(-1, 1))**2
            norm_sum_p += np.linalg.norm(u_test[Nx*Ny:, i])**2
            
        
        metrics = {
            'relative_error_total': np.sqrt(error_sum) / np.sqrt(norm_sum),
            'relative_error_q': np.sqrt(error_sum_q) / np.sqrt(norm_sum_q),
            'relative_error_p': np.sqrt(error_sum_p) / np.sqrt(norm_sum_p),
        }
        

        return metrics
    