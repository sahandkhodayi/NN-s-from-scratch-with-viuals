import numpy as np 




def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def leaky_relu(x):
    return np.where(x > 0, x, 0.01 * x)


def tanh(x):
    return np.tanh(x)


def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


ACTIVATIONS = {
    "sigmoid": sigmoid,
    "relu": relu,
    "leaky_relu": leaky_relu,
    "tanh": tanh,
    "softmax": softmax
}



class NEURAL:

    def __init__(self,number_of_last_conects,bias=0,weight=None,activation=None):

        self.num=number_of_last_conects
        self.weight=weight
        self.bias=bias
        self.last_input=None
        self.output=None
        self.activation=activation
        self.z=None


    def calculate_val(self,input):
        if self.num!=len(self.weight):
            raise ValueError("length of weight does'nt match the number of inputs")
        
        val=input@self.weight + self.bias
        self.last_input=input
        self.z=val
        val=self.activate(val)
        self.output=val
        return val


    def activate(self,z):
        if self.activation in ACTIVATIONS:
            func=ACTIVATIONS.get(self.activation)
            return func(z)
        else:
            self.activation=None
            return z
        
