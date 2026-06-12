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
        
        val=np.dot(input, self.weight) + self.bias
        self.last_input=list(input)
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

    def backward(self, grad_from_next):
        """
        grad_from_next : scalar gradient arriving from the next layer (dLoss/dOutput)
        Returns dLoss/dInput (list) to pass back to the previous layer.
        """
        from derivative import DERIVATIVES

        # 1. Multiply by activation derivative to get delta
        if self.activation in DERIVATIVES:
            act_deriv = DERIVATIVES[self.activation](self.output)
        else:
            act_deriv = 1  # linear / no activation → derivative is 1

        delta = grad_from_next * act_deriv   # dLoss/dZ

        # 2. Gradients for weights and bias
        self.dW = [delta * xi for xi in self.last_input]  # dLoss/dW
        self.dB = delta                                     # dLoss/dB

        # 3. Gradient to pass back to previous layer (dLoss/dInput per input)
        d_input = [delta * w for w in self.weight]
        return d_input

