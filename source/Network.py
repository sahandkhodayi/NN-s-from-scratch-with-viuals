import numpy as np
from layer import LAYERS
from neuron import NEURAL as node

class Network:
    
    def __init__(self,layers:list[LAYERS],)-> None:
        self.layers=layers
        self.output=None


    def predict(self,x:list[float])->list[float]:

        output=x
            
        for index in (self.layers):
            output=index.forward(output)
        self.output=output
        return output    

                
                