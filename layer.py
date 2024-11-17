from Neuron import neuron

import numpy as np

class layer_nuron:
    def __init__(self, nuron_amount=16):
        self.layer= []
        for i in range(nuron_amount):
            self.layer.append(neuron(nuron_amount))

class layer_none:
    def __init__(self,nuron_amount=16,connection_amount=16):
        self.weights=np.random.randn(nuron_amount,connection_amount)
        self.bias=np.random.randn(connection_amount)