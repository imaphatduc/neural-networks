import numpy as np


def compute_cost(AL, Y):
    loss = -(np.dot(Y, np.log(AL).T) + np.dot((1 - Y), np.log(1 - AL).T))

    m = Y.shape[1]

    cost = (1 / m) * loss

    cost = np.squeeze(cost)

    assert(cost.shape == ())

    return cost
