#!/usr/bin/env python

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



def wave2D(self):
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