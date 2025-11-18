#!/usr/bin/env python

import numpy as np

from .general_utilities import apply_decoder, get_jacobian
from pymor.vectorarrays.numpy import NumpyVectorSpace


def Galerkin_residuum(model, x, xn_1, mu, dt, fom, u_ref, scaled_data):
    """Manifold Galerkin residual for the implicit midpoint rule.

    ODE: \dot(x, \mu) = X_H(x; \mu)
    Residual r(x) := x - x_{n-1} - dt * X_H(x_mid),
    with x_mid = (x + x_{n-1})/2 and
    H(z) = J_g(z)^+ * M^{-1} * [-X_H(u_ref + g(z))].
    """
    # evaluate everything at the midpoint in reduced coordinates
    x_mid = 0.5 * (x + xn_1)

    # decode midpoint to full-order state and add reference
    decoded_mid = apply_decoder(x_mid, model, scaled_data)

    u_ref_1, u_ref_2 = np.split(u_ref, [int(u_ref.shape[0]/2)])
    decoded_mid_1, decoded_mid_2 = np.split(decoded_mid, [int(decoded_mid.shape[0]/2)])
    
   
    space = NumpyVectorSpace(model.dims[1]*model.dims[2])
    arr1 = space.from_numpy(u_ref_1 + decoded_mid_1)
    arr2 = space.from_numpy(u_ref_2 + decoded_mid_2)
    temp_mid = fom.operator.source.make_array([arr1, arr2])

    # RHS at midpoint
    # alternatively for Hamiltonian system: without minus and use J @ H_op instead of operator
    rhs_mid = (-1) * fom.operator.apply(temp_mid, mu=mu).to_numpy()

    # Jacobian of decoder at midpoint 
    # Moore–Penrose inverse of the jacobian at midpoint
    jac_mid = get_jacobian(model.network.decoder, x_mid, model, scaled_data).detach().numpy()
    mpr_jac_mid = np.linalg.pinv(jac_mid)
    prod_mid = (mpr_jac_mid @ rhs_mid).reshape(1,-1)

    # residual for implicit midpoint rule
    return x - xn_1 - dt * prod_mid


def Jacobian_approximate_Galerkin_residuum(model, x, xn_1, mu, dt, fom, u_ref, scaled_data):
    """Approximate Jacobian of the midpoint residual (Quasi-Newton).

    Using the same approximation as in the screenshot: keep only
    J_g^+ * M^{-1} * (∂F/∂u) * J_g, but evaluate at x_mid and
    include the chain factor 1/2 from x_mid = (x + x_{n-1})/2.
    """
    # midpoint
    x_mid = 0.5 * (x + xn_1)

    # decoded full state at midpoint
    decoded_mid = apply_decoder(x_mid, model, scaled_data)

    u_ref_1, u_ref_2 = np.split(u_ref, [int(u_ref.shape[0]/2)])
    decoded_mid_1, decoded_mid_2 = np.split(decoded_mid, [int(decoded_mid.shape[0]/2)])

    space = NumpyVectorSpace(model.dims[1]*model.dims[2])
    temp_mid = fom.operator.source.make_array([space.from_numpy(u_ref_1 + decoded_mid_1), space.from_numpy(u_ref_2 + decoded_mid_2)])
    
    operator_jac_mid = (-1) * fom.operator.jacobian(temp_mid, mu=mu)

    # J_g^+ * (∂F/∂u) * J_g at midpoint
    jac_mid = get_jacobian(model.network.decoder, x_mid, model, scaled_data).detach().numpy()
    mpr_jac_mid = np.linalg.pinv(jac_mid)
   
    jac_mid_1, jac_mid_2 = np.split(jac_mid, [int(jac_mid.shape[0]/2)])
    jac_mid = operator_jac_mid.source.make_array([space.from_numpy(jac_mid_1), space.from_numpy(jac_mid_2)])
    mpr_jac_mid_1, mpr_jac_mid_2 = np.split(mpr_jac_mid, [int(mpr_jac_mid.shape[1]/2)], axis=1)
    mpr_jac_mid = operator_jac_mid.source.make_array([space.from_numpy(mpr_jac_mid_1.T), space.from_numpy(mpr_jac_mid_2.T)])

    prod_mid = operator_jac_mid.apply2(mpr_jac_mid, jac_mid, mu=mu)

    # chain factor 1/2 because dr/dx has dH/dx_mid * d x_mid/dx, and d x_mid/dx = 1/2 I
    return np.eye(x_mid.shape[1]) - (dt * 0.5) * prod_mid


def Galerkin_line_search(model, x, p, xn_1, mu, dt, fom, u_ref, scaled_data, min_stepsize=5e-2, frac=0.9, c_1=1e-4, c_2=0.9):
    """Strong Wolfe line search."""
    alpha = 1.0

    res_orig = Galerkin_residuum(model, x, xn_1, mu, dt, fom, u_ref, scaled_data)
    #res_norm_orig = np.linalg.norm(res_orig)
    #p_times_grad_orig = np.inner(p, res_orig)[0,0]
    res_norm_orig = 0.5 * np.linalg.norm(res_orig)**2
    p_times_grad_orig = (-1) * np.linalg.norm(res_orig)**2
    
    res_update = Galerkin_residuum(model, x + alpha * p, xn_1, mu, dt, fom, u_ref, scaled_data)
    #res_norm_update = np.linalg.norm(res_update)
    #p_times_grad_update = np.inner(p, res_update)[0,0]
    J_approx_update = Jacobian_approximate_Galerkin_residuum(model, x + alpha * p, xn_1, mu, dt, fom, u_ref, scaled_data)
    res_norm_update = 0.5 * np.linalg.norm(res_update)**2
    p_times_grad_update = np.dot(p, J_approx_update.T @ res_update.T)[0,0]
    
    while (res_norm_update > res_norm_orig + c_1 * alpha * p_times_grad_orig
        or abs(p_times_grad_update) > c_2 * abs(p_times_grad_orig)):
        
        alpha *= frac 
        res_update = Galerkin_residuum(model, x + alpha * p, xn_1, mu, dt, fom, u_ref, scaled_data)

        #res_norm_update = np.linalg.norm(res_update)
        #p_times_grad_update = np.inner(p, res_update)[0,0]

        J_approx_update = Jacobian_approximate_Galerkin_residuum(model, x + alpha * p, xn_1, mu, dt, fom, u_ref, scaled_data)
        res_norm_update = 0.5 * np.linalg.norm(res_update)**2
        p_times_grad_update = np.dot(p, J_approx_update.T @ res_update.T)[0,0]

        #print(f'reduce alpha to: {alpha}')
        if alpha * frac < min_stepsize:
            print(" backtracking NOT successful. ")
            break
            
    #return alpha, res_update, res_norm_update
    return alpha, res_update, np.sqrt(2*res_norm_update)


def Galerkin_quasi_newton(model, xn_1, mu, dt, fom, u_ref, scaled_data, tol=1e-8):
    """Quasi-Newton with midpoint residual."""
    x_new = xn_1
    res = Galerkin_residuum(model, x_new, xn_1, mu, dt, fom, u_ref, scaled_data)
    #res_norm = 0.5 * np.linalg.norm(res)**2
    res_norm = np.linalg.norm(res)
    step = 0

    while res_norm > tol:
        J_approx = Jacobian_approximate_Galerkin_residuum(model, x_new, xn_1, mu, dt, fom, u_ref, scaled_data)
        p = np.linalg.solve(J_approx, -res.T).T

        alpha, res, res_norm = Galerkin_line_search(model, x_new, p, xn_1, mu, dt, fom, u_ref, scaled_data)
        x_new = x_new + alpha * p
        step += 1

        print(f'Step: {step} Residual norm: {res_norm}')

    return x_new
