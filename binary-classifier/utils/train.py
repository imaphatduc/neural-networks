from .compute_cost import compute_cost
from .linear_forward import L_model_forward
from .linear_backward import L_model_backward


def update_parameters(parameters, gradients, learning_rate):
    L = len(parameters) // 2

    parameters_updated = {}

    for l in range(L):
        parameters_updated["W" + str(l + 1)] = parameters["W" +
                                                          str(l + 1)] - learning_rate * gradients["dW" + str(l + 1)]

        parameters_updated["b" + str(l + 1)] = parameters["b" +
                                                          str(l + 1)] - learning_rate * gradients["db" + str(l + 1)]

    return parameters_updated


def train(X, Y, parameters, num_iterations, learning_rate, print_cost=False):
    costs = []

    for i in range(num_iterations):
        AL, caches = L_model_forward(X, parameters)

        cost = compute_cost(AL, Y)

        costs.append(cost)

        if print_cost and i % 100 == 0:
            print(f"Cost after {i} iterations: {cost}")

        gradients = L_model_backward(AL, Y, caches)

        parameters = update_parameters(parameters, gradients, learning_rate)

    return parameters, costs

