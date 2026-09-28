#!/usr/bin/env python

import numpy as np
import torch
import pickle


class Scaler:
    """Scaling and unscaling for 2-channel data (q, p)."""

    def __init__(self, dims, filename="./scaling/scaling_grid", eps=1e-12, device="cpu"):
        self.eps = eps
        self.device = device
        self.dims = dims
        self.Nx = dims[1]
        self.Ny = dims[2]

        with open(f'{filename}_{self.Nx}x{self.Ny}', "rb") as f:
            scaling = pickle.load(f)

        self.min_q = scaling["min"]["q"]
        self.min_p = scaling["min"]["p"]
        self.max_q = scaling["max"]["q"]
        self.max_p = scaling["max"]["p"]

        # Store ranges for convenience
        self.range_q = self.max_q - self.min_q + self.eps
        self.range_p = self.max_p - self.min_p + self.eps

    def scale(self, x):
        """Scale physical (q, p) data to [0, 1]."""
        #scaled = torch.empty_like(x, dtype=torch.double)
        scaled = np.empty_like(x)
        if x.ndim == 3:
            q, p = x[0, :, :], x[1, :, :]
            scaled[0, :, :] = (q - self.min_q) / self.range_q
            scaled[1, :, :] = (p - self.min_p) / self.range_p
        elif x.ndim == 4:
            q, p = x[:, 0, :, :], x[:, 1, :, :]
            scaled[:, 0, :, :] = (q - self.min_q) / self.range_q
            scaled[:, 1, :, :] = (p - self.min_p) / self.range_p
        else: 
            raise NotImplementedError
        return scaled

    def unscale(self, x):
        """Unscale from [0, 1] back to physical (q, p)."""
        #unscaled = torch.empty_like(x, dtype=torch.double)
        unscaled = np.empty_like(x)
        if x.ndim == 3:
            q, p = x[0, :, :], x[1, :, :]
            unscaled[0, :, :] = q * self.range_q + self.min_q
            unscaled[1, :, :] = p * self.range_p + self.min_p
        elif x.ndim == 4: 
            q, p = x[:, 0, :, :], x[:, 1, :, :]
            unscaled[:, 0, :, :] = q * self.range_q + self.min_q
            unscaled[:, 1, :, :] = p * self.range_p + self.min_p
        else: 
            raise NotImplementedError
        return unscaled
    
    
    def check_bounds(self, x_physical):
        """Check if test data is within training range."""
        if x_physical.ndim == 3:
            q, p = x_physical[0], x_physical[1]
        else:
            q, p = x_physical[:, 0], x_physical[:, 1]
            
        if np.min(q) < self.min_q or np.max(q) > self.max_q:
            print(f"WARNING: q values [{np.min(q):.4f}, {np.max(q):.4f}] "
                  f"outside training range [{self.min_q:.4f}, {self.max_q:.4f}]")
        if np.min(p) < self.min_p or np.max(p) > self.max_p:
            print(f"WARNING: p values [{np.min(p):.4f}, {np.max(p):.4f}] "
                  f"outside training range [{self.min_p:.4f}, {self.max_p:.4f}]")

  
    def unscale_and_prolongate_derivative(self, J):
        """
        Rescale Jacobian J according to physical units.

        Parameters
        ----------
        J : torch.Tensor
            Shape (C*H*W, L).
        dims : tuple
            (C, H, W).
        """
        C, H, W = self.dims
        L = J.shape[-1]

        if J.ndim == 3 and J.shape[1] == 1:
            J = J.squeeze(1)

        J = J.view(C, H, W, L)
        scales = torch.tensor([self.range_q, self.range_p], dtype=J.dtype, device=J.device).view(C, 1, 1, 1)
        J = J * scales
        return J.view(C * H * W, L)


    def prolongate_derivative(self, J):
        """
        Rescale Jacobian J according to physical units.

        Parameters
        ----------
        J : torch.Tensor
            Shape (C*H*W, L).
        dims : tuple
            (C, H, W).
        """
        C, H, W = self.dims
        L = J.shape[-1]

        if J.ndim == 3 and J.shape[1] == 1:
            J = J.squeeze(1)
        assert J.shape == (C * H * W, L), f"expected {(C*H*W, L)}, got {tuple(J.shape)}"

        return J.view(C * H * W, L)

    def restrict(self, x):
        """Reshape vector to (C, H, W)."""
        return x.reshape(self.dims[0], self.dims[1], self.dims[2])

    def prolongate(self, x):
        """Flatten field to column vector."""
        return x.reshape(-1,1)