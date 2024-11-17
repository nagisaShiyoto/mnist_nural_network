import numpy as np

class neuron:
    def __init__(self,connection_size=16):
        self.weights=np.random.randn(connection_size)
        self.bias=np.random.randn(connection_size)
    