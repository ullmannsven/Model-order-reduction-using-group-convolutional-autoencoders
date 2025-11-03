

class NoFilepathPassedError(Exception):
    pass


class InvalidNeuralNetworkError(Exception):
    def __init__(self):
        super().__init__("A neural networks needs at least two layers and " +
                         "all layers need a positive number of neurons!")
