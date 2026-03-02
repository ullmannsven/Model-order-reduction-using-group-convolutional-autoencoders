#!/usr/bin/env python

from abc import abstractmethod
import os

import torch
from equiv_networks.trainer import Trainer

class BaseModel:

    def __init__(self, network, dims, network_parameters={}, trainer=Trainer,
                 parameters_trainer={}):

        self.network = network(dims=dims, **network_parameters).double()
        self.trainer = trainer(self, **parameters_trainer)
        self.path = None

    def train(self, parameters_training={}):
        return self.trainer.train(**parameters_training)

    #TODO this is done differently on the server
    def save_neural_network(self, model_to_save=None, path=None):
        assert model_to_save is not None
        if path is None:
            if self.path is None:
                import time
                time_string = time.strftime("%Y%m%d-%H%M%S")
                path = "torch-model-" + time_string + ".pt"
                self.path = path
        else:
            self.path = path

        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        torch.save(model_to_save.network.state_dict(), self.path)

    def load_neural_network(self, path=None):
        if path is None:
            self.network.load_state_dict(torch.load(self.path))
        else:
            self.network.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
