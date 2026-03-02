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
    nt: int = 500
    timestep_factor: int = 1
     
    # Initial condition parameters
    x0: float = 0.5
    y0: float = 0.5
    sig: float = 0.05
    sig_pre: float = 0.5

    rotated: bool = False

    # visualization flag
    visualize_q: bool = True
      
    @property
    def hx(self):
        """Grid spacing in x-direction."""
        return self.Lx / self.Nx
    
    @property
    def hy(self):
        """Grid spacing in y-direction."""
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
    
    def __init__(self, config):
        """Initialize wave experiment.
        
        Args:
            config: Experiment configuration. If None, uses default values.
        """
        self.config = config
        self._fom = None
    
    @property
    def fom(self):
        """Create the full order model."""
        if self._fom is None:
            fom = self.build_wave_2D_fom()
            self._fom = fom
        return self._fom
        
    
    def _get_grid(self):
        """Get the spatial grid.
        
        Returns:
            Tuple of (x, y) coordinate arrays.
        """
        x = np.linspace(0, self.config.Lx, self.config.Nx, endpoint=False)
        y = np.linspace(0, self.config.Ly, self.config.Ny, endpoint=False)
        return x, y

    def _compute_derivative(self, Q, direction):
        """Compute spatial derivative with outflow boundary conditions.
        
        Uses one-sided derivatives at boundaries to implement outflow.
        
        Parameters:
            Q: 
                Array of shape (Ny, Nx) containing field values.
            direction: 
                'x' or 'y' for derivative direction.
            
        Returns:
            Derivative array of same shape as Q.
        """
        dQ = np.zeros_like(Q)
        
        if direction == 'x':
            h = self.config.hx
            # central difference with periodic wrap: roll left/right in x-direction
            dQ = (np.roll(Q, -1, axis=1) - np.roll(Q, 1, axis=1)) / (2 * h)
        elif direction == 'y':
            h = self.config.hy
            # central difference with periodic wrap: roll up/down in y-direction
            dQ = (np.roll(Q, -1, axis=0) - np.roll(Q, 1, axis=0)) / (2 * h)
        else:
            raise ValueError(f"Invalid direction: {direction}")
    
        return dQ
    

    def _get_initial_condition(self):
        """Compute initial condition (q0, p0) for diagonal pulse moving to upper-left.
        
        Pulse is along the diagonal y=x, moving upward-left (135 degrees).
        The initial condition extends to the boundaries without tapering.
        
        Returns:
            Tuple of (q0_flat, p0_flat) as 1D arrays.
        """ 
        x, y = self._get_grid()
        X, Y = np.meshgrid(x, y, indexing='xy')

        if not self.config.rotated:
            # Pulse along diagonal: distance from line y=x
            #dist_from_diag = (Y - X) / np.sqrt(2)
            d = (Y - X + 0.5) % 1.0 - 0.5
            dist_from_diag = d / np.sqrt(2.0)    
            
            # Gaussian pulse centered on the diagonal - extends to boundaries
            q0_flat = np.exp(-(dist_from_diag**2) / (2 * self.config.sig_pre * self.config.sig**2)).ravel()
            
            Q = q0_flat.reshape(self.config.Ny, self.config.Nx)
            
            # For 135-degree (upper-left) motion: velocity direction is (-1, 1) / sqrt(2)
            # p = -c * (-∂q/∂x + ∂q/∂y) / sqrt(2)
            dqdx_mat = self._compute_derivative(Q, direction='x')
            dqdy_mat = self._compute_derivative(Q, direction='y')
            
            # Combine derivatives for upper-left diagonal motion
            combined_deriv = (-dqdx_mat + dqdy_mat) / np.sqrt(2)

        else: 
            s = (X + Y - 1.0)
            d = (s + 0.5) % 1.0 - 0.5
            dist_from_diag = d / np.sqrt(2.0)

            q0_flat = np.exp(-(dist_from_diag**2) / (2 * self.config.sig_pre * self.config.sig**2)).ravel()

            Q = q0_flat.reshape(self.config.Ny, self.config.Nx)
            dqdx = self._compute_derivative(Q, direction='x')
            dqdy = self._compute_derivative(Q, direction='y')

            # n·∇q for n = (1, 1)/sqrt(2)
            combined_deriv = (dqdx + dqdy) / np.sqrt(2.0)

        p0_flat = combined_deriv.ravel()
        
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
            decoded_u_0_hat = model.scaler.prolongate(model.scaler.unscale(decoded_u_0_hat))
        else:
            decoded_u_0_hat = model.scaler.prolongate(decoded_u_0_hat)
        
        # Compute reference offset
        q0, p0 = self._get_initial_condition()
        initial_state = np.hstack((q0, - mu_val * p0)).reshape(-1, 1)
        u_ref = initial_state - decoded_u_0_hat
        
        return u_ref, initial_state, u_0_hat
    
    def build_wave_2D_fom(self):
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

        def D2_periodic_1d(n, h):
            """Second derivative with periodic boundary conditions in 1D."""
            main = -2.0 * np.ones(n)
            off = 1.0 * np.ones(n - 1)

            A = sp.diags([off, main, off], offsets=[-1, 0, 1], format='lil')

            # wrap-around entries for periodicity
            A[0, -1] = 1.0
            A[-1, 0] = 1.0

            return A.tocsr() / (h ** 2)
        
        Ix = sp.eye(Nx, format='csr')
        Iy = sp.eye(Ny, format='csr')

        # Laplacian with outflow BC on all sides: Δ = d^2/dx^2 + d^2/dy^2
        Dxx = D2_periodic_1d(Nx, hx)
        Dyy = D2_periodic_1d(Ny, hy)

        Lapl = (sp.kron(Iy, Dxx) + sp.kron(Dyy, Ix)).tocsr()
    
        # Hamiltonian operator H_op = diag(- c^2 K, I) on the block space
        Hqq = LincombOperator(operators=[NumpyMatrixOperator(Lapl)], coefficients=[(-1) * mu * mu])
        Hpp = NumpyMatrixOperator(sp.eye(Nx*Ny, format='csr'))
        H_op = BlockDiagonalOperator([Hqq, Hpp])

        q0, p0 = self._get_initial_condition()

        V = NumpyVectorSpace(Nx*Ny)

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
            model = QuadraticHamiltonianModel(T=T, initial_data=u0, H_op=H_op, visualizer=QBlockVisualizer(Nx=Nx, Ny=Ny), nt=nt, name='2D wave with outflow BC')
        else: 
            model = QuadraticHamiltonianModel(T=T, initial_data=u0, H_op=H_op, visualizer=PBlockVisualizer(Nx=Nx, Ny=Ny), nt=nt, name='2D wave with outflow BC')

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