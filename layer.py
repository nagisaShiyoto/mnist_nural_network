from Neuron import neuron

import numpy as np

class layer_nuron:
    def __init__(self, nuron_amount=16):
        self.layer= []
        for i in range(nuron_amount):
            self.layer.append(neuron(nuron_amount))

class layer_none:
    def __init__(self,nuron_amount=16,connection_amount=16):
        #create metrix of all the weghts to pass throw to that layer
        self.weights=np.random.randn(connection_amount,nuron_amount)
        #metrix shape(current amout X future amount)
        self.bias=np.random.randn(connection_amount)

    @staticmethod
    def sig(x):
        return 1/(1 + np.exp(-x))
    @staticmethod
    def relu(x):
        return max(0,x)
    
    def forward_propagation(self, nuron_layer):
        """create a dot multipycation of the nuron layer and the weights+ bias
        dot multypocation takes the first weight and multyply it by the first nuron
        the second on the second and this is how it goes threw every line of the merix
        every line is the weights we need to calc for the next nuron(so we aonly need to multyply it
        by the needed nuron)
        """
        raw_output=np.add(self.weights @ nuron_layer,self.bias)
        return self.sig(raw_output)