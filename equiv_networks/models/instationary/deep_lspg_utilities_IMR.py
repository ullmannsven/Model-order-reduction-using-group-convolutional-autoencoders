#!/usr/bin/env python

import numpy as np

from .general_utilities import apply_decoder, get_jacobian
from pymor.vectorarrays.numpy import NumpyVectorSpace


def LSPG_residuum(model, x, xn_1, mu, dt, fom):
    """Computes the manifold LSPG residual when using implicit midpoint rule as time stepping scheme."""
    #already high dimensional in this case
    x_mid = 0.5 * (x + xn_1)

    x_mid_1, x_mid_2 = np.split(x_mid, [int(x_mid.shape[0]/2)])
    
    space = NumpyVectorSpace(model.dims[1]*model.dims[2])
    temp_mid = fom.operator.source.make_array([space.from_numpy(x_mid_1), space.from_numpy(x_mid_2)])

    rhs = (-1) * fom.operator.apply(temp_mid, mu).to_numpy().reshape(-1,1)
    return x - xn_1 - dt * rhs


def Psi_matrix(model, x, xn_1, mu, dt, fom, u_ref, scaled_data):
    """Computes Petrov-Galerkin test matrix Psi."""
    jac = get_jacobian(model.network.decoder, x, model, scaled_data).detach().numpy()

    decoded = apply_decoder(x, model, scaled_data)
    decoded_xn = apply_decoder(xn_1, model, scaled_data)

    u_ref_1, u_ref_2 = np.split(u_ref, [int(u_ref.shape[0]/2)])
    decoded_1, decoded_2 = np.split(decoded, [int(decoded.shape[0]/2)])
    decoded_xn1, decoded_xn2 = np.split(decoded_xn, [int(decoded_xn.shape[0]/2)])
    
    from pymor.vectorarrays.numpy import NumpyVectorSpace
    space = NumpyVectorSpace(model.dims[1]*model.dims[2])
    temp_mid = 0.5 * fom.operator.source.make_array([space.from_numpy(2* u_ref_1 + decoded_1 + decoded_xn1), space.from_numpy(2 * u_ref_2 + decoded_2 + decoded_xn2)])

    # compute Jacobian of the right hand side (i.e. of the operator)
    operator_jac = (-1) * fom.operator.jacobian(temp_mid, mu=mu)
    # apply inverse of mass matrix to Jacobian of the right hand side
    #operator_jac = np.linalg.solve(fom.mass.matrix.array(), operator_jac)

    jac_1, jac_2 = np.split(jac, [int(jac.shape[0]/2)])
    jac = operator_jac.source.make_array([space.from_numpy(jac_1), space.from_numpy(jac_2)])

    return jac.to_numpy() - 0.5 * dt * operator_jac.apply(jac, mu=mu).to_numpy()
    


def LSPG_line_search(model, x, p, xn_1, mu, dt, fom, u_ref, scaled_data, min_stepsize=5e-2, frac=0.9, c_1=1e-4, c_2=0.9):
    """Performs line search according to Strong Wolfe conditions."""
    # initialize step size
    alpha = 1.0
    decoded = apply_decoder(x, model, scaled_data)
    decoded_xn_1 = apply_decoder(xn_1, model, scaled_data)

    # compute residual for current position
    res_orig = LSPG_residuum(model, u_ref + decoded, u_ref + decoded_xn_1, mu, dt, fom)
    Psi_mat = Psi_matrix(model, x, xn_1, mu, dt, fom, u_ref, scaled_data)
    #res_norm_orig = np.linalg.norm(Psi_mat.T @ res_orig)
    #p_times_grad_orig = p @ (Psi_mat.T @ res_orig)
    res_norm_orig = 0.5 * np.linalg.norm(Psi_mat.T @ res_orig)**2
    p_times_grad_orig = np.dot(p, Psi_mat.T @ Psi_mat @ Psi_mat.T @ res_orig)
    
    # compute decoded full-order representation of reduced coordinates given by x + alpha* p
    decoded = apply_decoder(x + alpha * p, model, scaled_data)
    res_update = LSPG_residuum(model, u_ref + decoded, u_ref + decoded_xn_1, mu, dt, fom)
    Psi_mat = Psi_matrix(model, x + alpha * p, xn_1, mu, dt, fom, u_ref, scaled_data)
    #res_norm_update = np.linalg.norm(Psi_mat.T @ res_update)
    #p_times_grad_update = p @ (Psi_mat.T @ res_update)
    res_norm_update = 0.5 * np.linalg.norm(Psi_mat.T @ res_update)**2
    p_times_grad_update = np.dot(p, Psi_mat.T @ Psi_mat @ Psi_mat.T @ res_update)

    # reduce step size as long as strong Wolfe conditions are not fulfilled
    while (res_norm_update > res_norm_orig + c_1 * alpha * p_times_grad_orig
          or abs(p_times_grad_update) > abs(c_2 * p_times_grad_orig)):
        
        alpha = alpha * frac
        decoded = apply_decoder(x + alpha * p, model, scaled_data)
        res_update = LSPG_residuum(model, u_ref + decoded, u_ref + decoded_xn_1, mu, dt, fom)
        Psi_mat = Psi_matrix(model, x + alpha * p, xn_1, mu, dt, fom, u_ref, scaled_data)
        res_norm_update = 0.5 * np.linalg.norm(Psi_mat.T @ res_update)**2
        p_times_grad_update = np.dot(p, Psi_mat.T @ Psi_mat @ Psi_mat.T @ res_update)
        #res_norm_update = np.linalg.norm(Psi_mat.T @ res_update)
        #p_times_grad_update = p @ (Psi_mat.T @ res_update)

        #print(f'reduce alpha to {alpha}')
    
        if alpha * frac < min_stepsize:
            break

    return alpha, res_update, np.sqrt(2*res_norm_update)


def LSPG_quasi_newton(model, xn_1, mu, dt, fom, u_ref, scaled_data, tol=1e-7, max_steps=100):
    """Performs Quasi-Newton iteration."""
    x_new = xn_1
    decoded = apply_decoder(x_new, model, scaled_data)

    #Note: This computes the residual without multiplication of the psi matrix
    res = LSPG_residuum(model, u_ref + decoded, u_ref + decoded, mu, dt, fom)
    res_norm = np.linalg.norm(res)

    step = 0
    p = [2*tol]
    # perform Quasi-Newton steps until norm of the residual has reached prescribed tolerance
    #not (np.linalg.norm(p) < tol)
    while (res_norm > tol and not (step >= max_steps)):
        Psi_mat = Psi_matrix(model, x_new, xn_1, mu, dt, fom, u_ref, scaled_data)
        res_disc = Psi_mat.T @ res
      
        PsiT_Psi = Psi_mat.T @ Psi_mat
        p = np.linalg.solve(PsiT_Psi, - res_disc).T
       
        alpha, res, res_norm = LSPG_line_search(model, x_new, p, xn_1, mu, dt, fom, u_ref, scaled_data)

        x_new = x_new + alpha * p
        step += 1

        print(f'Step: {step} Residual norm: {res_norm}')

    return x_new
