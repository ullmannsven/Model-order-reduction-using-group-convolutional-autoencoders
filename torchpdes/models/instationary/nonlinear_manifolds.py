#!/usr/bin/env python

import torch
import torch.nn as nn

from torchpdes.models.base import BaseModel
from torchpdes.training.trainer import Trainer


# class NonlinearManifoldsMOR1D(BaseModel):
#     """Class that implements the approach using nonlinear manifolds generated
#        by convolutional autoencoders for one-dimensional equations.

#     Parameters
#     ----------
#     full_order_model
#         Full order model that can be used to create snapshots for training/
#         validation/testing.
#     network
#         Constructor of the convolutional autoencoder to use.
#     network_parameters
#         Additional parameters used to create the neural network.
#     trainer
#           Trainer to use for the neural network.
#     parameters_trainer
#         Additional parameters used to create the Trainer.
#     data_creator
#         Object that creates training and validation samples.
#     loss_function
#         Loss function to use in the training of the neural network.
#     """

#     def __init__(self, 
#                  full_order_model,
#                  network=None,
#                  network_parameters={'encoder_channels': [8, 16, 32, 64],
#                                      'decoder_channels': [64, 32, 16, 8],
#                                      'max_pools_encoder': [False, False, True],
#                                      'unpool_decoder': [False, True],
#                                      'encoder_fully_connected_layers_sizes': [64*248, 256, 5],
#                                      'decoder_fully_connected_layers_sizes': [5, 256, 64*248],
#                                      'activation_function': nn.ELU()},
#                  trainer=Trainer, 
#                  parameters_trainer={}, 
#                  data_creator=None,
#                  loss_function=nn.MSELoss()):

#         self.full_order_model = full_order_model
        
#         super().__init__(network=network, network_parameters=network_parameters, trainer=trainer,
#                          parameters_trainer=parameters_trainer, data_creator=data_creator)

#         self.loss_function = loss_function
#         self.data_creator = data_creator
#         self.save_checkpoint = True

#     def prepare_batch(self, batch):
#         """Prepare the data for usage in training procedure (here we assume that
#            the solutions consist of a single quantity of interest, i.e. a single
#            channel for the training data).

#         Parameters
#         ----------
#         batch
#             Batch to convert.

#         Returns
#         -------
#         Dictionary with input and target tensors.
#         """
#         inputs = []
#         targets = []

#         for item in batch:
#             u_full_step_shifted = item['u_full_step_shifted']
#             inputs.append(u_full_step_shifted)
#             targets.append(u_full_step_shifted)

#         #Note: input and targets are the same, as we are training AE
#         return {'inputs': inputs, 'targets': targets}


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
