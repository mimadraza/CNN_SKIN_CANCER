import numpy as np

class FullyConnectedLayer:

    def __init__(self,input_size,num_nodes):
        self.input_size = input_size
        self.num_nodes = num_nodes

        self.weight = np.random.random((num_nodes,input_size))
        self.bias = np.random.random(num_nodes,1)

    def forward_prop(self,input):

        input_reshaped = input.reshape(-1, order='C')

        Z = np.dot(self.weight,input_reshaped) + self.bias
        a = np.max(Z,0)
        p = 1/1+np.exp(-a)

        return p
