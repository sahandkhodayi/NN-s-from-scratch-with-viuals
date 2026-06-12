import numpy as np



class MSE:

    def __init__(self):
        self.loss=None
        self.grad=None


    def forward(self,Y_true:list,Y_predict:list):
        diff=[]

        for i in range(len(Y_predict)):
            diff.append((Y_predict[i]-Y_true[i])**2)
        
        
        loss=(1/len(Y_predict))*sum(diff)
        self.loss=loss
        return loss
    
    def backward(self,Y_true:list,Y_predict:list):
        # Return a list of per-output gradients (one per output neuron)
        # instead of a single scalar, so each neuron gets its own signal
        grad=[]

        for i in range(len(Y_predict)):
            grad.append((2/len(Y_predict)) * (Y_predict[i] - Y_true[i]))

        self.grad=grad
        return grad

        


            