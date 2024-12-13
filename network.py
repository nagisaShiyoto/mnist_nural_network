from layer import layer_none as layer_class

class network:
    def __init__(self,layers_connection_amount=[784,16,16,10]):
        self.all_layers=[]
        for i,neuron_amount in enumerate(layers_connection_amount):
            if not i==0:
                #amount of last nurons= amount of connections
                connection_amount=layers_connection_amount[i-1]
                self.all_layers.append(layer_class(connection_amount,neuron_amount))

    
    def get_output(self,img):
        """
        get the output of the prediction.
        running in loop on every layer(which has the layers params-weghtts and biases)
        and use the result of the last layer as the nuron for the calculation
        in oher words:
        each time you calculate the next layer of nurons using the last one
        input:img- the first layer
        output: the last layer- the result
        """
        nuron_layer=img
        for i,layer in enumerate(self.all_layers):
            nuron_layer=layer.forward_propagation(nuron_layer)
        return nuron_layer
        

    
