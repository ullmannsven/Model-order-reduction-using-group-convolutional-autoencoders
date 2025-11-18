#!/usr/bin/env python

import torch
import torch.nn as nn

from torchpdes.models.base import BaseModel
from torchpdes.training.trainer import Trainer


class NonlinearManifoldsMOR2D(BaseModel):
    """Class that implements the approach using nonlinear manifolds generated
       by convolutional autoencoders for the two-dimensional reacting flow.

    Parameters
    ----------
    full_order_model
        Full order model that can be used to create snapshots for training/
        validation/testing.
    network
        Constructor of the convolutional autoencoder to use.
    network_parameters
        Additional parameters used to create the neural network.
    trainer
          Trainer to use for the neural network.
    parameters_trainer
        Additional parameters used to create the Trainer.
    data_creator
        Object that creates training and validation samples.
    loss_function
        Loss function to use in the training of the neural network.
    """

    def __init__(self, 
                 network=None,
                 scaler=None,
                 dims=None,
                 network_parameters=None,
                 trainer=Trainer,
                 parameters_trainer={}, 
                 loss_function=nn.MSELoss()):

        super().__init__(network=network, dims=dims, network_parameters=network_parameters, trainer=trainer,
                         parameters_trainer=parameters_trainer)
        
        self.scaler = scaler
        self.dims = dims
        self.loss_function = loss_function
        self.save_checkpoint = True

    def prepare_batch(self, batch):
        """Prepare the data for usage in training procedure

        Parameters
        ----------
        batch
            Batch to convert.

        Returns
        -------
        Dictionary with input and target tensors.
        """
        inputs = []
        targets = []

        for item in batch:
            u_full_step_shifted = item['u_full_step_shifted'] # this is size (2, Nx, Ny) in the case of a hamiltonian system
            # input and targets are the same, as we aim to learn the identity in AE training
            inputs.append(u_full_step_shifted)
            targets.append(u_full_step_shifted)

        return {'inputs': inputs, 'targets': targets}
