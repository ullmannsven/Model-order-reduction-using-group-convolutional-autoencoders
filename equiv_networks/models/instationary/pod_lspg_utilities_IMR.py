#!/usr/bin/env python

import numpy as np
from pymor.vectorarrays.numpy import NumpyVectorSpace


def LSPG_residuum(reduced_basis, x, xn_1, mu, dt, fom):
    """Computes the manifold LSPG residual when using implicit midpoint rule as time stepping scheme."""
    x_mid = 0.5 * (x + xn_1) #already high dimensional in this case

    x_mid_1, x_mid_2 = np.split(x_mid, [int(x_mid.shape[0]/2)])
    
    space = NumpyVectorSpace(int(reduced_basis.shape[0]/2))
    temp_mid = fom.operator.source.make_array([space.from_numpy(x_mid_1), space.from_numpy(x_mid_2)])

    # compute right hand side
    rhs = (-1) * fom.operator.apply(temp_mid, mu).to_numpy().reshape(-1,1)

    return x - xn_1 - dt * rhs


def Psi_matrix(reduced_basis, x, xn_1, mu, dt, fom, u_ref):
    """Computes Petrov-Galerkin test matrix Psi."""
    jac = reduced_basis
    decoded = reduced_basis @ x.T
    decoded_xn = reduced_basis @ xn_1.T

    u_ref_1, u_ref_2 = np.split(u_ref, [int(u_ref.shape[0]/2)])
    decoded_1, decoded_2 = np.split(decoded, [int(decoded.shape[0]/2)])
    decoded_xn1, decoded_xn2 = np.split(decoded_xn, [int(decoded_xn.shape[0]/2)])
    
    from pymor.vectorarrays.numpy import NumpyVectorSpace
    space = NumpyVectorSpace(int(reduced_basis.shape[0]/2))
    temp_mid = 0.5 * fom.operator.source.make_array([space.from_numpy(2* u_ref_1 + decoded_1 + decoded_xn1), space.from_numpy(2 * u_ref_2 + decoded_2 + decoded_xn2)])

    # compute Jacobian of the right hand side (i.e. of the operator)
    operator_jac = (-1) * fom.operator.jacobian(temp_mid, mu=mu)
    
    jac_1, jac_2 = np.split(jac, [int(jac.shape[0]/2)])
    jac = operator_jac.source.make_array([space.from_numpy(jac_1), space.from_numpy(jac_2)])

    return jac.to_numpy() - 0.5 * dt * operator_jac.apply(jac, mu=mu).to_numpy()
    


def LSPG_line_search(reduced_basis, x, p, xn_1, mu, dt, fom, u_ref, min_stepsize=5e-2, frac=0.5, c_1=1e-4, c_2=0.9):
    """Performs line search according to Strong Wolfe conditions."""
    # initialize step size
    alpha = 1.0
    decoded = reduced_basis @ x.T
    decoded_xn_1 = reduced_basis @ xn_1.T

    # compute residual for current position
    res_orig = LSPG_residuum(reduced_basis, u_ref + decoded, u_ref + decoded_xn_1, mu, dt, fom)
    Psi_mat = Psi_matrix(reduced_basis, x, xn_1, mu, dt, fom, u_ref)
    res_norm_orig = 0.5 * np.linalg.norm(Psi_mat.T @ res_orig)**2
    p_times_grad_orig = np.dot(p, Psi_mat.T @ Psi_mat @ Psi_mat.T @ res_orig)
    
    # compute decoded full-order representation of reduced coordinates given by x + alpha* p
    decoded = reduced_basis @ (x + alpha * p).T
    res_update = LSPG_residuum(reduced_basis, u_ref + decoded, u_ref + decoded_xn_1, mu, dt, fom)
    Psi_mat = Psi_matrix(reduced_basis, x + alpha * p, xn_1, mu, dt, fom, u_ref)
    res_norm_update = 0.5 * np.linalg.norm(Psi_mat.T @ res_update)**2
    p_times_grad_update = np.dot(p, Psi_mat.T @ Psi_mat @ Psi_mat.T @ res_update)

    # reduce step size as long as strong Wolfe conditions are not fulfilled
    while (res_norm_update > res_norm_orig + c_1 * alpha * p_times_grad_orig
          or abs(p_times_grad_update) > abs(c_2 * p_times_grad_orig)):
        
        alpha = alpha * frac
        decoded = reduced_basis @ (x + alpha*p).T
        res_update = LSPG_residuum(reduced_basis, u_ref + decoded, u_ref + decoded_xn_1, mu, dt, fom)
        Psi_mat = Psi_matrix(reduced_basis, x + alpha * p, xn_1, mu, dt, fom, u_ref)
        res_norm_update = 0.5 * np.linalg.norm(Psi_mat.T @ res_update)**2
        p_times_grad_update = np.dot(p, Psi_mat.T @ Psi_mat @ Psi_mat.T @ res_update)

        #print(f'reduce alpha to {alpha}')
    
        if alpha * frac < min_stepsize:
            break

    return alpha, res_update, np.sqrt(2*res_norm_update), True

import torch

def POD_LSPG_quasi_newton(reduced_basis, xn_1, mu, dt, fom, u_ref, tol=1e-7, max_steps=100):
    """Performs Quasi-Newton iteration."""
    x_new = xn_1
    decoded = reduced_basis @ x_new.T
    
    #Note: This computes the residual without multiplication of the psi matrix
    res = LSPG_residuum(reduced_basis, u_ref + decoded, u_ref + decoded, mu, dt, fom)
    res_norm = np.linalg.norm(res)

    step = 0
    p = [2*tol]
    lambda_reg = 0
    # perform Quasi-Newton steps until norm of the residual has reached prescribed tolerance
    while (res_norm > tol and not (np.linalg.norm(p) < tol) and not (step >= max_steps)):
        Psi_mat = Psi_matrix(reduced_basis, x_new, xn_1, mu, dt, fom, u_ref)
        res_disc = Psi_mat.T @ res
        
        PsiT_Psi = Psi_mat.T @ Psi_mat
        p = np.linalg.solve(PsiT_Psi, - res_disc).T
       
        alpha, res, res_norm, success = LSPG_line_search(reduced_basis, x_new, p, xn_1, mu, dt, fom, u_ref)

        x_new = x_new + alpha * p
        step += 1

        print(f'Step: {step} Residual norm: {res_norm}')

    return x_new

