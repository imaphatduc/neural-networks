from .linear_forward import L_model_forward


def predict(X, parameters):
    AL, caches = L_model_forward(X, parameters)

    Y_prediction = (AL >= 0.5) * 1.0

    return Y_prediction
