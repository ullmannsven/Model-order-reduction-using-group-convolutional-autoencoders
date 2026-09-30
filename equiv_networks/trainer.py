#!/usr/bin/env python

import torch
import torch.utils as utils
import torch.nn as nn
from torch.amp import autocast, GradScaler 

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
    parameters_lr_scheduler
        Additional parameters for the learning rate scheduler.
    use_validation
        Determines whether or not to perform a validation phase.
    es_scheduler
        Early stopping scheduler to use.
    parameters_es_scheduler
        Additional parameters for the early stopping scheduler.
    device
        Device on which the training in performed. Either "cpu" or "cuda".
    loss_mode
        Determines the loss function to use for training. If "physical", the loss is computed in physical units (i.e. after unscaling the outputs and targets). 
        If "weights", a weighted MSE loss is used, where the weights are computed as the inverse of the variance of each channel in the targets. 
        If "symplectic", the loss is a weighted sum of the data loss (computed in physical units) and a symplectic loss that penalizes deviations from 
        the symplectic condition on the Jacobian of the decoder.
        If None, the loss function defined in the model is used without modification.
    targets_are_normalized
        Only relevant if loss_mode == "physical". Determines whether or not the targets are normalized (i.e. need to be unscaled) before computing the loss in physical units.
    loss_symplectic_fraction
        Only relevant if loss_mode == "symplectic". Determines the fraction of the data loss in the overall loss (the rest is the symplectic loss).
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
        self.gradscaler = GradScaler() if self.device.type == "cuda" else None
        self.global_step = 0

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

        inputs = [b['u_full_step_shifted'] for b in batch]
        if not isinstance(inputs[0], torch.Tensor):
            inputs = [torch.as_tensor(x) for x in inputs]

        inputs = torch.stack(inputs, dim=0)

        return {'inputs': inputs, 'targets': inputs}


    def train(self, training_data=None, training_loader=None,
              number_of_epochs=1000, batch_size=20, learning_rate=None, 
              validation_data=None, validation_loader=None, show_progress_bar=True, nn_save_filepath=None):
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
        nn_save_filepath
            Filepath where to save the biases and weigths of the trained NN.
       
        Returns
        -------
        Minimum validation and training loss (if early stopping scheduler is used).
        """
        if learning_rate is not None:
            for group in self.optimizer.param_groups:
                group['lr'] = learning_rate

        print()
        print('Training of neural network:')
        print('===========================')
        print()
        
        print("which device", self.device) 
        if self.device.type == "cuda":
            workers = 8
        else: 
            workers = 0
        use_workers = workers > 0

        if not training_loader:
            training_sampler = utils.data.RandomSampler(training_data)
            training_loader = utils.data.DataLoader(training_data,
                                                    batch_size=batch_size,
                                                    collate_fn=lambda batch: self.prepare_batch(batch),
                                                    sampler=training_sampler, 
                                                    pin_memory=(self.device.type == "cuda"), 
                                                    num_workers=workers, 
                                                    persistent_workers=use_workers)

       
        if self.use_validation and not validation_loader:
            validation_sampler = None
            validation_loader = utils.data.DataLoader(validation_data, 
                                                      batch_size=batch_size,
                                                      collate_fn=lambda batch: self.prepare_batch(batch),
                                                      sampler=validation_sampler, 
                                                      pin_memory=(self.device.type == "cuda"), 
                                                      num_workers=workers,
                                                      persistent_workers=use_workers)

        return self.train_network(training_loader, 
                                  number_of_epochs=number_of_epochs,
                                  learning_rate=learning_rate,
                                  validation_loader=validation_loader,
                                  show_progress_bar=show_progress_bar,
                                  nn_save_filepath=nn_save_filepath)
    
    

    def _compute_loss(self, outputs, targets, inputs, encoded_inputs=None):
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
        
        elif self.loss_mode == "symplectic":
            data_loss = self.model.loss_function(outputs, inputs)
            symplectic_loss = self.symplectic_loss_chunked(self.model.network.decoder, encoded_inputs)
           
            return self.loss_symplectic_fraction * data_loss + (1 - self.loss_symplectic_fraction) * symplectic_loss
        
        else: 
            return self.model.loss_function(outputs, targets)
        
    def symplectic_loss_chunked(self, decoder, encoded_inputs, chunk_size=5):
        losses = []
        for chunk in encoded_inputs.split(chunk_size):
            losses.append(self.symplectic_loss_optimized(decoder, chunk))
        return torch.stack(losses).mean()
    

    def symplectic_loss_optimized(self, decoder, encoded_inputs):
        B, r = encoded_inputs.shape
        device, dtype = encoded_inputs.device, encoded_inputs.dtype
            
        # Use reverse-mode AD (better for many outputs, few inputs)
        J_single = torch.func.jacfwd(lambda z: self.decoder_flat(decoder, z))
        J_batch = torch.vmap(J_single)(encoded_inputs)  # (B, 2N, r)
        J = torch.vmap(lambda J_b: self.model.scaler.unscale_and_prolongate_derivative(J_b))(J_batch)
      
	    # Compute O @ J efficiently without materializing O
        # For J: (B, 2N, r), split into top/bottom halves
        N =  J.shape[1] // 2
        J_top = J[:, :N, :]   # q components
        J_bot = J[:, N:, :]   # p components
    
        # O @ J = [J_bot; -J_top] (exploit block structure)
        OmegaJ = torch.cat([J_bot, -J_top], dim=1)  # (B, 2N, r)
    
        # Compute S = J^T @ (O @ J) via batched matmul
        S = torch.bmm(J.transpose(1, 2), OmegaJ)  # (B, r, r)
    
        # Create canonical Poisson tensor O_ri
        n = r // 2
        Om_r = torch.zeros((r, r), device=device, dtype=dtype)
        Om_r[:n, n:] = torch.eye(n, device=device, dtype=dtype)
        Om_r[n:, :n] = -torch.eye(n, device=device, dtype=dtype)
    
        # Compute loss
        diff = S - Om_r.unsqueeze(0)  # (B, r, r)
        loss = (diff.pow(2).sum(dim=(-2, -1)) / (r * r)).mean()
    
        return loss
    

    def decoder_flat(self, decoder, z):
        squeeze_back = False
        if z.dim() == 1:
            z = z.unsqueeze(0)
            squeeze_back = True

        y = decoder(z) 
        y = y.flatten(start_dim=1)

        if squeeze_back: 
            y = y.squeeze(0)
        return y


    def weighted_mse(self, pred_phys, target_phys, w):
        """
        pred_phys, target_phys: (B, C, H, W) in physical units
        w: scalar, list/tuple, or 1D tensor of length C (per-channel weights)
        """
        w = torch.as_tensor(w, device=pred_phys.device, dtype=pred_phys.dtype)
        w = w.view(1, -1, 1, 1)
        return ((pred_phys - target_phys).pow(2) * w).mean()
    

    def train_network(self, training_loader, number_of_epochs=1000,
                      learning_rate=None, validation_loader=None, show_progress_bar=True, nn_save_filepath=None):
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
        nn_save_filepath
            Filepath where to save the biases and weigths of the trained NN.
       
        Returns
        -------
        Minimum validation and training loss (if early stopping
        scheduler is used).
        """
        
        number_of_training_samples = len(training_loader.dataset)
        number_of_validation_samples = 0
        if validation_loader:
            number_of_validation_samples = len(validation_loader.dataset)

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
            print("HALLO welche epoch", epoch, self.device) 
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
                    
                    inputs  = batch['inputs'].to(self.device, non_blocking=(self.device.type=="cuda"))
                    targets = batch['targets'].to(self.device, non_blocking=(self.device.type=="cuda"))        

                    with torch.set_grad_enabled(phase == 'train'):
                        with autocast("cuda", enabled=(self.device.type == "cuda")):
                            outputs = self.model.network(inputs)
                            # compute loss
                            encoded_inputs = None
                            if self.loss_mode == "symplectic":
                                encoded_inputs = self.model.network.encoder(inputs)

                            loss = self._compute_loss(outputs, targets, inputs, encoded_inputs=encoded_inputs)

                            # additional loss terms computed inside the network's forward pass (e.g. pose loss)
                            auxiliary_loss = getattr(self.model.network, "auxiliary_loss", None)
                            if callable(auxiliary_loss):
                                loss = loss + auxiliary_loss()

                            # perform step of optimizer if in training phase
                            if phase == 'train':
                                self.optimizer.zero_grad()

                                if self.gradscaler is not None:
                                    self.gradscaler.scale(loss).backward()
                                    self.gradscaler.step(self.optimizer)
                                    self.gradscaler.update()
                                    self.global_step += 1
                                else:
                                    loss.backward()
                                    self.optimizer.step()
                                    self.global_step += 1


                    # update current loss
                    running_loss += loss.item() * len(batch["inputs"])
        
            
                # update loss in current epoch
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                losses[phase] = epoch_loss
 
                # check if early stopping is possible
                if phase == 'val' and self.es_scheduler and self.es_scheduler(losses['val'], losses['train'], self.model.save_checkpoint):
                    print()
                    print('Early stopping...')
                    print(f'Minimum validation loss: {self.es_scheduler.best_loss}')
                    
                    return self.es_scheduler.best_loss, self.es_scheduler.training_loss
 
            #print training and validation losses after each epoch
            if show_progress_bar:
                if self.use_validation:
                    bar.update(losses['train'], losses['val'])
                else:
                    bar.update(losses['train'])

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
        number_of_training_samples:
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

