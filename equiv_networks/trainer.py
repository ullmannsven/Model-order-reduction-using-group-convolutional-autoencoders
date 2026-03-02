#!/usr/bin/env python

import os
import numpy as np

import torch
import torch.utils as utils
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

class ProgressTraining:

    def __init__(self, total, prefix='', suffix='', decimals=1, print_end='\r'):
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self.decimals = decimals
        self.iteration = 0
        self.print_end = print_end

    def update(self, training_loss=None, validation_loss=None):
        percent = ("{0:." + str(self.decimals) + "f}").format(100 * (self.iteration / float(self.total)))
        if training_loss is None:
            print('\r{} {}% {}'.format(self.prefix, percent, self.suffix), end=self.print_end)
        elif validation_loss is None:
            print('\r{} {}% {}; TL: {:.4E}'.format(self.prefix, percent, self.suffix, training_loss), end=self.print_end)
        else:
            print('\r{} {}% {}; TL: {:.4E}; VL: {:.4E}'.format(self.prefix, percent, self.suffix, training_loss, validation_loss), end=self.print_end)

        if self.iteration == self.total:
            print()
        self.iteration = self.iteration + 1


class Trainer:
    """Class that implements a generic trainer for neural networks.

    Parameters
    ----------
    model
        Model for that the respective neural network should be trained.
    optimizer
        Optimizer to use for training.
    parameter_optimizer
        Additional parameters for the optimizer.
    learning_rate
        Initial default learning rate for the optimizer.
    lr_scheduler
        Learning rate scheduler to use.
    use_validation
        Determines whether or not to perform a validation phase.
    es_scheduler
        Early stopping scheduler to use.
    parameters_es_scheduler
        Additional parameters for the early stopping scheduler.
    device
        Device on which the training in performed. Either "cpu" or "cuda".
        #TODO hier fehlen noch einige input arguments
    """
    def __init__(self, model, optimizer=None, parameters_optimizer={}, learning_rate=None,
                 lr_scheduler=None, parameters_lr_scheduler=None, use_validation=True,
                 es_scheduler=None, parameters_es_scheduler={}, device=None, 
                 loss_mode=None, targets_are_normalized=True, loss_symplectic_fraction=None):
        
        if learning_rate:
            self.learning_rate = learning_rate

        if es_scheduler:
            es_scheduler = es_scheduler(self, **parameters_es_scheduler)
        self.es_scheduler = es_scheduler

        self.use_validation = use_validation

        self.device = (torch.device(device) if device is not None 
                       else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        self.model = model
        self.model.network.to(self.device)

        if optimizer:
            self.optimizer = optimizer(self.model.network.parameters(), lr=learning_rate, **parameters_optimizer)

        if lr_scheduler:
            lr_scheduler = lr_scheduler(self.optimizer, **parameters_lr_scheduler)
        self.lr_scheduler = lr_scheduler

        self.loss_mode = loss_mode
        self.targets_are_normalized = targets_are_normalized

        if self.loss_mode == "symplectic": 
            assert loss_symplectic_fraction
            self.loss_symplectic_fraction = loss_symplectic_fraction


    def train(self, training_data=None, training_loader=None,
              number_of_epochs=1000, batch_size=20, learning_rate=None, 
              validation_data=None, validation_loader=None, show_progress_bar=True,
              log_filename=None, nn_save_filepath=None, log_frequency=10):
        """Sets up everything and call function to start training procedure.

        Parameters
        ----------
        training_data
            Already computed training data (if available).
        training_loader
            Data loader holding the training data and performing
            the (random) mini-batching of the training data.
        number_of_epochs
            Maximum number of training epochs to perform.
        batch_size
            Batch size to use for mini-batching (has to be the size of the
            training set when using L-BFGS as optimizer).
        learning_rate
            Initial learning rate for the each of the parameter groups of the
            optimizer.
        validation_data
            Already compute validation data (if available).
        validation_loader
            Data loader holding the validation data and performing
            the (random) mini-batching of the validation data.
        show_progress_bar
            Determines whether or not to show a progress bar during training.
        log_filename
            Filename for the logging file.
        nn_save_filepath
            Filepath where to save the biases and weigths of the trained NN.
        log_frequency
            Number of epochs to wait before updating the logging file.

        Returns
        -------
        Minimum validation and training loss (if early stopping
        scheduler is used).
        """
        # set learning rate for each parameter group in the optimizer to the given initial value
        if learning_rate is not None:
            for group in self.optimizer.param_groups:
                group['lr'] = learning_rate

        print()
        print('Training of neural network:')
        print('===========================')
        print()

        if self.device.type == "cuda":
            workers = 8
        else: 
            workers = 0
        use_workers = workers > 0

        if not training_loader:
            training_sampler = utils.data.RandomSampler(training_data)
            training_loader = utils.data.DataLoader(training_data,
                                                    batch_size=batch_size,
                                                    collate_fn=lambda batch: self.model.prepare_batch(batch),
                                                    sampler=training_sampler, 
                                                    pin_memory=(self.device.type == "cuda"), 
                                                    num_workers=workers, 
                                                    persistent_workers=use_workers)

       
        # define validation loader if necessary
        if self.use_validation and not validation_loader:
            validation_sampler = None
            validation_loader = utils.data.DataLoader(validation_data, 
                                                      batch_size=batch_size,
                                                      collate_fn=lambda batch: self.model.prepare_batch(batch),
                                                      sampler=validation_sampler, 
                                                      pin_memory=(self.device.type == "cuda"), 
                                                      num_workers=workers,
                                                      persistent_workers=use_workers)

        # train the neural network
        return self.train_network(training_loader, 
                                  number_of_epochs=number_of_epochs,
                                  learning_rate=learning_rate,
                                  validation_loader=validation_loader,
                                  show_progress_bar=show_progress_bar, 
                                  log_filename=log_filename,
                                  nn_save_filepath=nn_save_filepath,
                                  log_frequency=log_frequency)
    

    def _compute_loss(self, outputs, targets, encoded_inputs=None):
        """
        Compute loss according to self.loss_mode.
        Assumptions:
            - outputs, targets: shape [B, C, ...]
            - If loss_mode == "physical":
                - self.scaler.denorm(tensor) available
                - targets_are_normalized tells whether to denorm targets
            - If loss_mode == "weighted":
                - self.registered_channel_weights is a 1D tensor of length C
        """
        if self.loss_mode == "physical":
            y_phys = self.model.scaler.unscale(outputs)
            if self.targets_are_normalized:
                x_phys = self.model.scaler.unscale(targets)
            else:
                x_phys = targets
            return self.model.loss_function(y_phys, x_phys)
        
        elif self.loss_mode == "weights":
            var_b = targets.var(dim=(0,2,3), unbiased=False).detach()
            w = 1.0 / (var_b + 1e-8)
            w = 2.0 * w / w.sum()
            return self.weighted_mse(outputs, targets, w)
        
        #TODO this is outdated and done more efficient on the server
        elif self.loss_mode == "symplectic":
            data_loss = self.model.loss_function(outputs, targets)
            sum_part = 0.0
            for i in range(encoded_inputs.shape[0]):
                jac = self.get_jacobian(self.model.network.decoder, encoded_inputs[i, :]).detach().numpy()
                sum_part += np.linalg.norm(jac.T @ self.symplectic_poisson_tensor(10201) @ jac - self.symplectic_poisson_tensor(60))**2

            symplectic_loss = sum_part / (encoded_inputs.shape[1]**2 * outputs.shape[0]) #outputs.shape[0] is the batchsize, encoded_inputs.shape[1] is the reduced dimension

            return self.loss_symplectic_fraction * data_loss + (1 - self.loss_symplectic_fraction) * symplectic_loss
        
        else: 
            return self.model.loss_function(outputs, targets)
    
        
    def symplectic_poisson_tensor(self, n, dtype=float):
        """
        Return the canonical (constant) Poisson tensor J for R^{2n} with coords (q,p):
            J = [[0,  I_n],
                [-I_n, 0]]
        """
        I = np.eye(n, dtype=dtype)
        Z = np.zeros((n, n), dtype=dtype)
        J = np.block([[Z,  I], [-I, Z]])
        return J
        
    def get_jacobian(self, function, x):
        x = x.reshape(1,-1)
        x.requires_grad_(True)

        def f_latent(x):
            y = function(x)
            return y.reshape(-1)

        # Compute Jacobian d f / d x at x
        J = torch.func.jacfwd(f_latent)(x)
        dims = (2, 101, 101)

        return self.model.scaler.unscale_and_prolongate_derivative(J, dims)
    
    def weighted_mse(self, pred_phys, target_phys, w):
        """
        pred_phys, target_phys: (B, C, H, W) in physical units
        w: scalar, list/tuple, or 1D tensor of length C (per-channel weights)
        """
        w = torch.as_tensor(w, device=pred_phys.device, dtype=pred_phys.dtype)
        w = w.view(1, -1, 1, 1)
        return ((pred_phys - target_phys).pow(2) * w).mean()
    

    def train_network(self, training_loader, number_of_epochs=1000,
                      learning_rate=None, validation_loader=None, show_progress_bar=True,
                      log_filename=None, nn_save_filepath=None, log_frequency=10):
        """Performs actual training of the neural network.

        Parameters
        ----------
        training_loader
            Data loader holding the training data and performing
            the (random) mini-batching of the training data.
        number_of_epochs
            Maximum number of training epochs to perform.
        learning_rate
            Initial learning rate for the each of the parameter groups of the
            optimizer (here only required for printing of the parameters).
        validation_loader
            Data loader holding the validation data and performing
            the (random) mini-batching of the validation data.
        show_progress_bar
            Determines whether or not to show a progress bar during training.
        log_filename
            Filename for the logging file.
        nn_save_filepath
            Filepath where to save the biases and weigths of the trained NN.
        log_frequency
            Number of epochs to wait before updating the logging file.

        Returns
        -------
        Minimum validation and training loss (if early stopping
        scheduler is used).
        """
        if log_filename:
            os.makedirs(os.path.dirname(log_filename), exist_ok=True)
            writer = SummaryWriter(log_dir=f'{log_filename}')
            print('To see training progress on tensorboard, run:')
            print(f'tensorboard --logdir={os.path.dirname(log_filename)}')
            print()

        number_of_training_samples = len(training_loader.dataset)
        number_of_validation_samples = 0
        if validation_loader:
            number_of_validation_samples = len(validation_loader.dataset)

        # print training and network parameters
        self.print_parameters(number_of_training_samples, number_of_epochs,
                              training_loader, learning_rate=learning_rate,
                              number_of_validation_samples=number_of_validation_samples,
                              validation_loader=validation_loader)

        if show_progress_bar:
            bar = ProgressTraining(number_of_epochs, prefix='Train the network:', suffix='epochs completed')

        phases = ['train']
        dataloaders = {'train':  training_loader}

        if self.use_validation:
            phases.append('val')
            dataloaders['val'] = validation_loader

        # perform actual training iteration
        for epoch in range(number_of_epochs):
            losses = {}
            for phase in phases: # eiter 'train' or 'val'
                # set state of network according to current phase (training or validation)
                if phase == 'train':
                    self.model.network.train()
                else:
                    self.model.network.eval()

                running_loss = 0.0

                # iterate over all batches in the respective phase
                for batch in dataloaders[phase]:
                    # get inputs and targets
                    #NOTE this is done differently on the server
                    inputs  = torch.stack(batch['inputs']).to(self.device, non_blocking=(self.device.type=="cuda"))
                    targets = torch.stack(batch['targets']).to(self.device, non_blocking=(self.device.type=="cuda"))        

                    with torch.set_grad_enabled(phase == 'train'):
                        
                        # define closure (especially for optimizers like L-BFGS this is required)
                        def closure():
                            if torch.is_grad_enabled():
                                self.optimizer.zero_grad()
                            # get outputs to current inputs with current network weights and biases
                            outputs = self.model.network(inputs)
                           
                            # compute loss
                            loss = self._compute_loss(outputs, targets)

                            # back propagate loss if necessary
                            if loss.requires_grad:
                                loss.backward()
                            # return loss
                            return loss

                        # perform step of optimizer if in training phase
                        if phase == 'train':
                            self.optimizer.step(closure)

                        # perform step of learning rate scheduler if necessary
                        if self.lr_scheduler and phase == 'train':
                            self.lr_scheduler.step()

                        # compute current loss
                        loss = closure()

                    # update current loss
                    running_loss += loss.item() * len(batch["inputs"])
        
                # log learning rate if necessary
                if self.lr_scheduler and log_filename and epoch % log_frequency == 0:
                    writer.add_scalar('average_learning_rate',
                                      np.average(self.lr_scheduler.get_lr()),
                                      global_step=epoch)

                # update loss in current epoch
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                losses[phase] = epoch_loss

                # perform validation
                if phase == 'val':
                    # log losses
                    if log_filename and epoch % log_frequency == 0:
                        writer.add_scalars(f'losses',
                                           {'training_loss': losses['train'],
                                            'validation_loss': losses['val']},
                                           global_step=epoch)

                    # check if early stopping is possible
                    if self.es_scheduler and self.es_scheduler(losses['val'], losses['train'], self.model.save_checkpoint):
                        print()
                        print('Early stopping...')
                        print(f'Minimum validation loss: {self.es_scheduler.best_loss}')
                        if log_filename:
                            writer.close()

                        return self.es_scheduler.best_loss, self.es_scheduler.training_loss

            # log training loss if necessary
            if not self.use_validation and log_filename and epoch % log_frequency == 0:
                writer.add_scalar('training_loss', losses['train'], global_step=epoch)
            
            #print training and validation losses after each epoch
            if show_progress_bar:
                if self.use_validation:
                    bar.update(losses['train'], losses['val'])
                else:
                    bar.update(losses['train'])

        if log_filename:
            writer.close()

        if self.es_scheduler:
            return self.es_scheduler.best_loss, self.es_scheduler.training_loss
        else:
            return None

    def print_parameters(self, number_of_training_samples, number_of_epochs,
                         training_loader, learning_rate=None, number_of_validation_samples=0,
                         validation_loader=None):
        """Prints the parameters of the neural network and the training.

        Parameters
        ----------
        number_of_training_samples
            Number of training samples used during training the network.
        number_of_epochs
            Maximum number of training epochs to perform.
        training_loader
            Data loader holding the training data and performing
            the (random) mini-batching of the training data.
        learning_rate
            Initial learning rate for the each of the parameter groups of the
            optimizer.
        number_of_validation_samples
            Number of validation samples used during training the network.
        validation_loader
            Data loader holding the validation data and performing
            the (random) mini-batching of the validation data.
        """

        # print the parameters of the neural network
        self.model.network.print_parameters()

        print()

        print('=> Training parameters:')
        print(f'Training samples: {number_of_training_samples}')
        print(f'Epochs: {number_of_epochs}')
        print(f'Batch size: {training_loader.batch_size}')
        print(f'Training loader: {training_loader.__class__.__name__}')
        print(f'Mini-batch sampler: {training_loader.sampler.__class__.__name__}')
        print(f'Optimizer: {self.optimizer.__class__.__name__}')
        if learning_rate is not None:
            print(f'Initial learning rate: {learning_rate}')
        else:
            print(f'Initial learning rate: {self.learning_rate}')

        print()

        if self.use_validation:
            print('=> Validation parameters:')
            print(f'Validation samples: {number_of_validation_samples}')
            print(f'Batch size: {validation_loader.batch_size}')
            print(f'Validation loader: {validation_loader.__class__.__name__}')
            print(f'Mini-batch sampler: {validation_loader.sampler.__class__.__name__}')
            if self.es_scheduler:
                print(f'Early stopping scheduler: {self.es_scheduler.__class__.__name__}')
                if hasattr(self.es_scheduler, 'patience'):
                    print(f'Patience of early stopping scheduler: {self.es_scheduler.patience}')
                if hasattr(self.es_scheduler, 'maximum_loss'):
                    print(f'Maximum loss to stop with: {self.es_scheduler.maximum_loss}')
            else:
                print('No early stopping used')
        else:
            print('=> No validation phase used')

        print()
