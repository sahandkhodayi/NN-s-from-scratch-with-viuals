import numpy as np


def sigmoid_derivative(output):
    return output * (1 - output)


def relu_derivative(output):
    if output > 0:
        return 1
    else:
        return 0
    

def leaky_relu_derivative(output):
    if output>0:
        return 1
    else:
        return 0.01


def tanh_derivative(output):
    return 1 - output**2



DERIVATIVES = {
    "sigmoid": sigmoid_derivative,
    "relu": relu_derivative,
    "leaky_relu": leaky_relu_derivative,
    "tanh": tanh_derivative,
    
}
