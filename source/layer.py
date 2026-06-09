import numpy as np 
from neuron import NEURAL as node

class LAYERS:

    def __init__(self,nodes:list[node],num_nodes:int=0,num_input:int=0)->None:
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