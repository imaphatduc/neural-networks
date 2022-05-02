import numpy as np
from .activation import relu, sigmoid


def linear_forward(A, W, b):
    Z = np.dot(W, A) + b

    assert(Z.shape == (W.shape[0], A.shape[1]))

    cache = A, W, b

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    if activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)


    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)


    assert (A.shape == (W.shape[0], A_prev.shape[1]))

    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters):
    L = len(parameters) // 2

    caches = []

    A = X

    for l in range(1, L):
        A_prev = A

        Wl = parameters["W" + str(l)]
        bl = parameters["b" + str(l)]

        A, cache = linear_activation_forward(A_prev, Wl, bl, "relu")

        caches.append(cache)


    WL = parameters["W" + str(L)]
    bL = parameters["b" + str(L)]

    AL, cache = linear_activation_forward(A, WL, bL, "sigmoid")

    assert(AL.shape == (1, X.shape[1]))

    caches.append(cache)

    return AL, caches
