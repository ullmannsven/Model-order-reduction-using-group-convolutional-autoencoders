#!/usr/bin/env python

import numpy as np
import torch

from .general_utilities import apply_decoder, get_jacobian


def Galerkin_residuum(model, x, xn_1, mu, dt, fom, u_ref):
    """Manifold Galerkin residual for the implicit midpoint rule.

    ODE: \dot(x, \mu) = X_H(x; \mu)
    Residual r(x) := x - x_{n-1} - dt * X_H(x_mid),
    with x_mid = (x + x_{n-1})/2 and
    H(z) = J_g(z)^+ * M^{-1} * [-X_H(u_ref + g(z))].
    """
    # evaluate everything at the midpoint in reduced coordinates
    x_mid = 0.5 * (x + xn_1)

    # decode midpoint to full-order state and add reference
    decoded_mid = apply_decoder(x_mid, model)
    #TODO this is only due to the blockstructure of our operator
    u_ref_1, u_ref_2 = np.split(u_ref, [int(u_ref.shape[0]/2)])
    decoded_mid_1, decoded_mid_2 = np.split(decoded_mid, [int(decoded_mid.shape[0]/2)])
    
    from pymor.vectorarrays.numpy import NumpyVectorSpace
    space = NumpyVectorSpace(model.dims[1]*model.dims[2])
    arr1 = space.from_numpy(u_ref_1 + decoded_mid_1)
    arr2 = space.from_numpy(u_ref_2 + decoded_mid_2)
    temp_mid = fom.operator.source.make_array([arr1, arr2])

    # RHS at midpoint
    # alternatively for Hamiltonian system: without minus and use J @ H_op instead of operator
    rhs_mid = (-1) * fom.operator.apply(temp_mid, mu=mu).to_numpy()

    #rhs_mid = rhs_mid_phys.copy()
    #rhs_mid[:10201] /= den_q
    #rhs_mid[10201:] /= den_p

    # apply M^{-1}  (keep as in your code)
    #rhs_mid = np.linalg.solve(fom.mass.matrix.array(), rhs_mid.to_numpy())
    #TODO needs to be done differently

    #Test jacobian of decoder
    # x_test = torch.randn(1, 120, dtype=torch.float64)
    # jac_test = get_jacobian(model.network.decoder, x_test, model).detach().numpy()

    # print("waas", x_test.requires_grad)

    # with torch.no_grad():
    #     h = 1e-5 * (1.0 + x_test.abs())
    #     h = h.view(1, 120)  # [1, D]

    # jac_fd = np.zeros((20402, 120), dtype=np.float64)
    # for i in range(120):
    #     ei = torch.zeros_like(x_test)
    #     ei[0, i] = 1.0

    #     hi = h[0, i].item()

    #     x_plus  = x_test + hi * ei
    #     x_minus = x_test - hi * ei

    #     y_plus  = apply_decoder(x_plus, model)
    #     y_minus = apply_decoder(x_minus, model)

    #     col = ((y_plus - y_minus) / (2.0 * hi))
    #     jac_fd[:, i] = col.reshape(-1)

    # # ---------- metrics ----------
    # def safe_rel_err(a, b, eps=1e-12):
    #     na = np.linalg.norm(a)
    #     nb = np.linalg.norm(b)
    #     denom = max(na, nb, eps)
    #     return np.linalg.norm(a - b) / denom

    # fro_rel_err  = safe_rel_err(jac_fd, jac_test)
    # max_abs_diff = np.max(np.abs(jac_fd - jac_test))
    # per_col_rel  = [safe_rel_err(jac_fd[:, i], jac_test[:, i]) for i in range(120)]

    # print("Jacobian check @ midpoint x_test_12")
    # print(f"  Frobenius relative error : {fro_rel_err:.3e}")
    # print(f"  Max abs difference       : {max_abs_diff:.3e}")
    # print("  Per-column relative error:", ", ".join(f"{e:.2e}" for e in per_col_rel))


    # Jacobian of decoder at midpoint 
    # Moore–Penrose inverse of the jacobian at midpoint
    jac_mid = get_jacobian(model.network.decoder, x_mid, model).detach().numpy()
    mpr_jac_mid = np.linalg.pinv(jac_mid)
    #mpr_jac_mid = jac_mid.T
    prod_mid = (mpr_jac_mid @ rhs_mid).reshape(1,-1)

    # residual for implicit midpoint rule
    return x - xn_1 - dt * prod_mid


def Jacobian_approximate_Galerkin_residuum(model, x, xn_1, mu, dt, fom, u_ref):
    """Approximate Jacobian of the midpoint residual (Quasi-Newton).

    Using the same approximation as in the screenshot: keep only
    J_g^+ * M^{-1} * (∂F/∂u) * J_g, but evaluate at x_mid and
    include the chain factor 1/2 from x_mid = (x + x_{n-1})/2.
    """
    # midpoint
    x_mid = 0.5 * (x + xn_1)

    # decoded full state at midpoint
    decoded_mid = apply_decoder(x_mid, model)

    #TODO again this code is very shitty
    u_ref_1, u_ref_2 = np.split(u_ref, [int(u_ref.shape[0]/2)])
    decoded_mid_1, decoded_mid_2 = np.split(decoded_mid, [int(decoded_mid.shape[0]/2)])

    from pymor.vectorarrays.numpy import NumpyVectorSpace
    space = NumpyVectorSpace(model.dims[1]*model.dims[2])
    temp_mid = fom.operator.source.make_array([space.from_numpy(u_ref_1 + decoded_mid_1), space.from_numpy(u_ref_2 + decoded_mid_2)])
    
    # operator Jacobian (∂F/∂x) at midpoint
    #TODO i am not sure if there needs to be a minus here
    operator_jac_mid = (-1) * fom.operator.jacobian(temp_mid, mu=mu)

    # apply M^{-1}
    #TODO check if this is automaltically the id if it is not set
    #operator_jac_mid = np.linalg.solve(fom.mass.matrix.array(), operator_jac_mid)

    # J_g^+ * (∂F/∂u) * J_g at midpoint
    jac_mid = get_jacobian(model.network.decoder, x_mid, model).detach().numpy()
    mpr_jac_mid = np.linalg.pinv(jac_mid)
   
    jac_mid_1, jac_mid_2 = np.split(jac_mid, [int(jac_mid.shape[0]/2)])
    jac_mid = operator_jac_mid.source.make_array([space.from_numpy(jac_mid_1), space.from_numpy(jac_mid_2)])
    mpr_jac_mid_1, mpr_jac_mid_2 = np.split(mpr_jac_mid, [int(mpr_jac_mid.shape[1]/2)], axis=1)
    mpr_jac_mid = operator_jac_mid.source.make_array([space.from_numpy(mpr_jac_mid_1.T), space.from_numpy(mpr_jac_mid_2.T)])

    prod_mid = operator_jac_mid.apply2(mpr_jac_mid, jac_mid, mu=mu)

    # chain factor 1/2 because dr/dx has dH/dx_mid * d x_mid/dx, and d x_mid/dx = 1/2 I
    return np.eye(x_mid.shape[1]) - (dt * 0.5) * prod_mid


def Galerkin_line_search(model, x, p, xn_1, mu, dt, fom, u_ref, min_stepsize=0.01, frac=0.9, c_1=1e-4, c_2=0.9):
    """Strong Wolfe line search."""
    alpha = 1.0

    res_orig = Galerkin_residuum(model, x, xn_1, mu, dt, fom, u_ref)
    res_norm_orig = np.linalg.norm(res_orig)
    p_times_grad_orig = np.inner(p, res_orig)[0,0]
    #res_norm_orig = 0.5 * np.linalg.norm(res_orig)**2
    #p_times_grad_orig = (-1) * np.linalg.norm(res_orig)**2 #NOTE this only works for 0 stepsize, so for the starting point"
    
    res_update = Galerkin_residuum(model, x + alpha * p, xn_1, mu, dt, fom, u_ref)
    res_norm_update = np.linalg.norm(res_update)
    p_times_grad_update = np.inner(p, res_update)[0,0]
    #J_approx_update = Jacobian_approximate_Galerkin_residuum(model, x + alpha * p, xn_1, mu, dt, fom, u_ref)
    #res_norm_update = 0.5 * np.linalg.norm(res_update)**2
    #p_times_grad_update = np.dot(p, J_approx_update.T @ res_update.T)[0,0]
    
    while (res_norm_update > res_norm_orig + c_1 * alpha * p_times_grad_orig
        or abs(p_times_grad_update) > c_2 * abs(p_times_grad_orig)):
        
        #print("why not successful", res_norm_update, res_norm_orig + c_1 * alpha * p_times_grad_orig)
        #print("or maybe this", abs(p_times_grad_update), c_2 * abs(p_times_grad_orig))

        alpha *= frac 
        res_update = Galerkin_residuum(model, x + alpha * p, xn_1, mu, dt, fom, u_ref)

        res_norm_update = np.linalg.norm(res_update)
        p_times_grad_update = np.inner(p, res_update)[0,0]

        #J_approx_update = Jacobian_approximate_Galerkin_residuum(model, x + alpha * p, xn_1, mu, dt, fom, u_ref)
        #res_norm_update = 0.5 * np.linalg.norm(res_update)**2
        #p_times_grad_update = np.dot(p, J_approx_update.T @ res_update.T)[0,0]

        print(f'reduce alpha to: {alpha}')
        if alpha * frac < min_stepsize:
            print(" backtracking NOT successful. ")
            break
            #J_approx_update = Jacobian_approximate_Galerkin_residuum(model, x + alpha * p, xn_1, mu, dt, fom, u_ref)
            #p = np.linalg.solve(J_approx_update, -res_update.T).T 
            
            #alpha = 1
            #res_update = Galerkin_residuum(model, x + alpha * p, xn_1, mu, dt, fom, u_ref)
            #res_norm_update = np.linalg.norm(res_update)
            #p_times_grad_update = np.inner(p, res_update)[0,0]
            #print("Warning: Did not find a proper step width. Reset alpha and compute new step p")
            
    
    return alpha, res_update, res_norm_update


# Armijo backtracking without using wolfe conditions
def backtracking(x, p, xn_1, mu, dt, fom, u_ref, alpha=1.0, frac=0.5, c=1e-4, max_it=40):
    
    r0 = Galerkin_residuum(x, xn_1, mu, dt, fom, u_ref)
    f0 = np.linalg.norm(r0)

    for _ in range(max_it):
        r1 = Galerkin_residuum(x + alpha*p, xn_1, mu, dt, fom, u_ref)
        f1 = np.linalg.norm(r1)
        if f1 <= (1 - c*alpha) * f0:
            return alpha, r1, f1
        alpha *= frac
    return alpha, r1, f1


def Galerkin_quasi_newton(model, xn_1, mu, dt, fom, u_ref, tol=1e-6):
    """Quasi-Newton with midpoint residual."""
    x_new = xn_1
    res = Galerkin_residuum(model, x_new, xn_1, mu, dt, fom, u_ref)
    #res_norm = 0.5 * np.linalg.norm(res)**2
    res_norm = np.linalg.norm(res)

    while res_norm > tol:
        J_approx = Jacobian_approximate_Galerkin_residuum(model, x_new, xn_1, mu, dt, fom, u_ref)
        p = np.linalg.solve(J_approx, -res.T).T

        alpha, res, res_norm = Galerkin_line_search(model, x_new, p, xn_1, mu, dt, fom, u_ref)
        x_new += alpha * p
 
        print(f'Residual norm: {res_norm}')

    return x_new
