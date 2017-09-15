import numpy as np

def relu(z, d = False):
    if d:
        return np.round(np.maximum(0, z))
    else:
        return np.maximum(0, z)

def sigmoid(z, d = False):
    if d:
        a = sigmoid(z)
        return a * (1 - a)
    else:
        return 1 / (1 + np.exp(-z))

def linear(z, d = False):
    if d:
        return 1
    else:
        return z
