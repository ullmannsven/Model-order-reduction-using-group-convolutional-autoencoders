#!/usr/bin/env python

"""Centralized experiment setup for 2D wave equation experiments."""

import numpy as np
import torch
from dataclasses import dataclass

from pymor.basic import *
from torchpdes.pdes.instationary import wave_2D

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
    
    # Physical parameters
    sig_pre: float = 0.5
    
    # Initial condition parameters
    x0: float = 0.3
    y0: float = 0.3
    sig: float = 0.05
    
    # Direction of flow
    x_flow: bool = True
    visualize_q: bool = True
      
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
        return wave_2D(T=self.config.T, Nx=self.config.Nx, Ny=self.config.Ny, sig_pre=self.config.sig_pre, nt=self.config.nt, x_flow=self.config.x_flow, visualize_q=self.config.visualize_q)
        
    # @property
    # def mu_test(self):
    #     """Get or create the test parameter."""
    #     if self._mu_test is None:
    #         self._mu_test = self.fom.parameters.parse({'mu': self.config.mu_test_val})
    #     return self._mu_test
    
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
            q0_flat = np.exp(-((X - self.config.x0)**2) / (2 * self.config.sig_pre * self.config.sig**2)).ravel()
            
            Q = q0_flat.reshape(self.config.Ny, self.config.Nx)
            dqdx_mat = self.compute_derivative(Q, direction='x_flow')
            dqdx_flat = dqdx_mat.ravel()
            p0_flat = - mu_val * dqdx_flat
        else:
            # Gaussian pulse in y-direction
            q0_flat = np.exp(-((Y - self.config.y0)**2) / (2 * self.config.sig_pre * self.config.sig**2)).ravel()
            
            Q = q0_flat.reshape(self.config.Ny, self.config.Nx)
            dqdy_mat = self.compute_derivative(Q, direction='y_flow')
            dqdy_flat = dqdy_mat.ravel()
            p0_flat = - mu_val * dqdy_flat
        
        return (q0_flat, p0_flat)
    
    def get_initial_state(self, mu_val):
        """Get full initial state as concatenated array.
        
        Returns:
            Array of shape (2*Nx*Ny, 1) containing [q0; p0].
        """
        q0, p0 = self._get_initial_condition(mu_val=mu_val)
        return np.hstack((q0, p0)).reshape(-1, 1)
    
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
        initial_state = self.get_initial_state(mu_val=mu_val)
        u_ref = initial_state - decoded_u_0_hat
        
        return u_ref, u_0_hat
    
    def solve_fom(self, mu):
        """Solve the full order model.
        
        Returns:
            Solution array of shape (2*Nx*Ny, n_timesteps).
        """
        #if mu is None:
        #    return self.fom.solve(self.mu_test)
        return self.fom.solve(mu)
    
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
     
    def compute_error_metrics(self, u_approx_full, u_test, u_approx_latent=None, model=None):
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
        
        latent_error = 0.0
        latent_error_norm = 0.0
        
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
            
            # Latent errors (if provided)
            if u_approx_latent is not None and model is not None:
                sol = torch.as_tensor(model.scaler.restrict(u_test[:, i] - u_test[:, 0]), dtype=torch.double, device="cpu").unsqueeze(0)
                sol_enc = model.network.encode(sol).detach().cpu().numpy()
                
                latent_error += np.linalg.norm(sol_enc.reshape(-1, 1) - u_approx_latent[i].reshape(-1, 1))**2
                latent_error_norm += np.linalg.norm(sol_enc)**2
        
        metrics = {
            'relative_error_total': np.sqrt(error_sum) / np.sqrt(norm_sum),
            'relative_error_q': np.sqrt(error_sum_q) / np.sqrt(norm_sum_q),
            'relative_error_p': np.sqrt(error_sum_p) / np.sqrt(norm_sum_p),
        }
        
        if u_approx_latent is not None and model is not None:
            metrics['relative_error_latent'] = (
                np.sqrt(latent_error) / np.sqrt(latent_error_norm)
            )
        
        return metrics