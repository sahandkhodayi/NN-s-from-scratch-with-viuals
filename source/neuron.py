import numpy as np 


# Activation function for non-linearity
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



# activation main HASHMAP!
ACTIVATIONS = {
    "sigmoid": sigmoid,
    "relu": relu,
    "leaky_relu": leaky_relu,
    "tanh": tanh,
    "softmax": softmax
}



#main class 
class NEURAL:

    def __init__(self,number_of_last_conects,bias=0,weight=None,activation=None):
        """
        This class is mainly about neurals and their behaviar like:
        Output= W.X+b
        and we apply the output to an activation.

        then we store it to its attribute for back probpagation
        Attribute:
        Weight: connection between previouse and main layer
        Bias
        num: length of neural
        Output: z with activation
        z:raw value
        """
        
    
        
        self.num=number_of_last_conects
        self.weight = weight if weight is not None else (np.random.randn(number_of_last_conects) * np.sqrt(2.0 / number_of_last_conects)).tolist()
        self.bias=bias
        self.last_input=None
        self.output=None
        self.activation=activation
        self.z=None


    def calculate_val(self,input:list[float])->float:
        """
        this function will use weights and last layers neurals to this layers new neurals

        Input (arr):
            last layers list or arr of neourals

        Return (float):
            new neurals output compare with the weights and last layers of neurals        
        
        """
        
        
        if self.num!=len(self.weight):
            raise ValueError("length of weight does'nt match the number of inputs")
        
        val=np.dot(input, self.weight) + self.bias
        self.last_input=list(input)
        self.z=val
        val=self.activate(val)
        self.output=val
        return val


    def activate(self,z:float)->float:
        """
        this function is used to change linearity of the outputs 

        args:
        z  (float):
            last output of the dot product between layers and weights

        output  (float):
            new float number that is not linear!    
        
        
        """
        

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

        
        if self.activation in DERIVATIVES:
            act_deriv = DERIVATIVES[self.activation](self.output)
        else:
            act_deriv = 1 

        delta = grad_from_next * act_deriv   # dLoss/dZ

       
        self.dW = [delta * xi for xi in self.last_input]  # dLoss/dW
        self.dB = delta                                     # dLoss/dB

        d_input = [delta * w for w in self.weight]
        return d_input