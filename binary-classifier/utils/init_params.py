import numpy as np


def initialize_parameters(layer_dims):
    L = len(layer_dims)

    parameters = {}

    for l in range(1, L):
        W = np.random.randn(
            layer_dims[l], layer_dims[l - 1]) / np.sqrt(layer_dims[l - 1])

        b = np.zeros((layer_dims[l], 1))

        parameters["W" + str(l)] = W
        parameters["b" + str(l)] = b

        assert(parameters['W' + str(l)].shape ==
               (layer_dims[l], layer_dims[l - 1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters
