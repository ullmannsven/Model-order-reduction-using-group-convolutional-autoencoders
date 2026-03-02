#!/usr/bin/env python

import torch


def apply_decoder(x, model, scaled_data):
    """Performs the forward pass of x through the whole decoder."""
    decoded = model.network.decode(torch.as_tensor(x, dtype=torch.double, device="cpu"))[0].detach().numpy()
    if scaled_data:
        decoded = model.scaler.unscale(decoded)
    decoded = model.scaler.prolongate(decoded)
    return decoded


def get_jacobian(function, x, model, scaled_data):
    # """Computes the Jacobian of function with respect to the inputs at point x."""
    
    x = torch.as_tensor(x, dtype=torch.double, device="cpu")
    x.requires_grad_(True)

    def f_latent(x):
        y = function(x)
        return y.reshape(-1)

    # Compute Jacobian d f / d x at x
    J = torch.func.jacfwd(f_latent)(x)

    if scaled_data:
        return model.scaler.unscale_and_prolongate_derivative(J)
    
    return model.scaler.prolongate_derivative(J)
