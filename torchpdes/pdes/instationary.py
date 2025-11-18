#!/usr/bin/env python

from pymor.basic import *

from pymor.operators.constructions import VectorOperator
from pymor.parameters.base import Parameters
from pymor.models.symplectic import QuadraticHamiltonianModel
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.block import BlockVectorSpace
from pymor.operators.block import BlockOperator
from pymor.discretizers.builtin.gui.visualizers import PatchVisualizer

import numpy as np
import scipy.sparse as sp

from pymor.models.symplectic import QuadraticHamiltonianModel
from pymor.vectorarrays.numpy import NumpyVectorSpace

from pymor.operators.numpy import NumpyMatrixOperator
from pymor.operators.constructions import LincombOperator
from pymor.operators.block import BlockDiagonalOperator, BlockColumnOperator
from pymor.parameters.functionals import ExpressionParameterFunctional



def wave_2D(T=1, Nx=101, Ny=101, nt=1000, sig_pre=None, x_flow=True, visualize_q=True):
    """ discrezized linear wave equation in 2D posed on [0,1]^2 as a Hamiltonian system

    Parameters
    T

    Nx

    Ny 

    x_flow


    Returns

    """
  
    assert sig_pre

    Lx = Ly = 1.0
  
    if x_flow:
        hx = Lx/Nx
        hy = Ly/(Ny-1)
    else: 
        hx = Lx/(Nx-1)
        hy = Ly/Ny
    
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
    if x_flow:
        Dxx = D2_periodic_1d(Nx, hx)
        Dyy = D2_neumann_1d(Ny, hy)
    else: 
        Dxx = D2_neumann_1d(Nx, hx)
        Dyy = D2_periodic_1d(Ny, hy)

    Lapl = (sp.kron(Iy, Dxx) + sp.kron(Dyy, Ix)).tocsr()
 
    # Hamiltonian operator H_op = diag(c^2 K, I) on the block space ---
    Hqq = LincombOperator(operators=[NumpyMatrixOperator(Lapl)], coefficients=[(-1) * mu * mu])
    Hpp = NumpyMatrixOperator(sp.eye(Nx*Ny, format='csr'))
    H_op = BlockDiagonalOperator([Hqq, Hpp])

    if x_flow:
        x = np.linspace(0, Lx, Nx, endpoint=False)  # periodic x
        y = np.linspace(0, Ly, Ny, endpoint=True)   # Neumann y
    else:
        x = np.linspace(0, Lx, Nx, endpoint=True)   # Neumann x
        y = np.linspace(0, Ly, Ny, endpoint=False)  # periodic y

    X, Y = np.meshgrid(x, y, indexing='xy')

    # initial data: Gaussian pulse in q with matched momentum \delta_t u = p = -c ∂_x q (pure right mover)
    if x_flow:
        x0 = 0.3
        sig = 0.05
        q0_flat = np.exp(-((X - x0)**2) / (2*sig_pre*sig**2)).ravel()
        
        Q = q0_flat.reshape(Ny, Nx)
        dqdx_mat = np.zeros_like(Q)
        dqdx_mat[:, 1:-1] = (Q[:, 2:] - Q[:, :-2])/(2*hx)
        dqdx_mat[:, 0] = (Q[:, 1] - Q[:, -1])/(2*hx)
        dqdx_mat[:, -1] = (Q[:, 0] - Q[:, -2])/(2*hx)
        dqdx_flat = dqdx_mat.ravel()

        #dqdx_flat = - 2 * (X - x0).ravel() / (2* sig_pre * sig**2) * q0_flat
        #dqdx_flat = D1 @ q0_flat

    else: 
        y0 = 0.3
        sig = 0.05
        q0_flat = np.exp(-((Y - y0)**2) / (2*sig_pre*sig**2)).ravel()

        Q = q0_flat.reshape(Ny, Nx)
        dqdy_mat = np.zeros_like(Q)
        dqdy_mat[1:-1, :] = (Q[2:, :] - Q[:-2, :])/(2*hy)
        dqdy_mat[0, :] = (Q[1, :] - Q[-1, :])/(2*hy)
        dqdy_mat[-1, :] = (Q[0, :] - Q[-2, :])/(2*hy)
        dqdy_flat = dqdy_mat.ravel()

    V = NumpyVectorSpace(Nx*Ny)

    if x_flow:
        u0 = BlockColumnOperator([VectorOperator(V.from_numpy(q0_flat)), LincombOperator([VectorOperator(V.from_numpy(dqdx_flat))], [(-1) * mu])])
    else: 
        u0 = BlockColumnOperator([VectorOperator(V.from_numpy(q0_flat)), LincombOperator([VectorOperator(V.from_numpy(dqdy_flat))], [(-1) * mu])])

    # visualizer that extract the q = u component of the solution
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
                q_flat = ui[: self.Nx * self.Ny]
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
                p_flat = ui[self.Nx * self.Ny :]  # p comes after q
                data.append(self.scalar_space.from_numpy(p_flat))

            return self.base.visualize(data, **kwargs)
        
    # build the Hamiltonian model
    nt = int(T * nt)
    if visualize_q:
        model = QuadraticHamiltonianModel(T=T, initial_data=u0, H_op=H_op, visualizer=QBlockVisualizer(Nx=Nx, Ny=Ny), nt=nt, name='2D wave')
    else: 
        model = QuadraticHamiltonianModel(T=T, initial_data=u0, H_op=H_op, visualizer=PBlockVisualizer(Nx=Nx, Ny=Ny), nt=nt, name='2D wave')

    return model
