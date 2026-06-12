import numpy as np 
from neuron import NEURAL as node

class LAYERS:

    def __init__(self,nodes:list[node],num_input)->None:
        self.main_nodes=nodes
        self.output=None
        self.num_inputs=num_input



    def forward(self,input)->list[float]:
        if self.num_inputs!=len(input):
            raise ValueError("length of input does'nt match!!!")
        
        output=[]
        
        
        for i in range(len(self.main_nodes)):
            output_node=self.main_nodes[i].calculate_val(input)
            output.append(output_node)
        self.output=output
        
        
        return output

    def backward(self, grads_from_next: list) -> list:
        """
        grads_from_next : list of gradients, one per neuron in THIS layer
                          (comes from the next layer or from the loss)
        Returns a list of gradients for the PREVIOUS layer's outputs.
        """
        # Each neuron returns a d_input list (one value per input feature).
        # We sum them up across all neurons to get the full gradient
        # that flows back to the previous layer.

        num_inputs = self.num_inputs
        grad_to_prev = [0.0] * num_inputs

        for i, neuron in enumerate(self.main_nodes):
            d_input = neuron.backward(grads_from_next[i])
            for j in range(num_inputs):
                grad_to_prev[j] += d_input[j]

        return grad_to_prev