import numpy as np
from .activation import relu_backward, sigmoid_backward


def linear_backward(dZ, cache):
    A_prev, W, b = cache

    m = A_prev.shape[1]

    dW = (1 / m) * np.dot(dZ, A_prev.T)

    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)

    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)

    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    gradients = {}

    L = len(caches)

    current_cache = caches[L - 1]

    Y = Y.reshape(AL.shape)

    # Element-wise operation, no need to transpose any of the two
    dAL = -np.divide(Y, AL) + np.divide(1 - Y, 1 - AL)

    gradients["dA" + str(L)] = dAL

    dAL_prev, dW, db = linear_activation_backward(
        dAL, current_cache, "sigmoid")

    gradients["dA" + str(L - 1)] = dAL_prev
    gradients["dW" + str(L)] = dW
    gradients["db" + str(L)] = db

    # l is the iterator of range [L - 2, 0]
    for l in reversed(range(L - 1)):
        current_cache = caches[l]

        dAl = gradients["dA" + str(l + 1)]

        # l here is the actual label of each layer in the nn, not the iterator above
        dAl_prev, dWl, dbl = linear_activation_backward(
            dAl, current_cache, "relu")

        gradients["dA" + str(l)] = dAl_prev
        gradients["dW" + str(l + 1)] = dWl
        gradients["db" + str(l + 1)] = dbl

    return gradients
