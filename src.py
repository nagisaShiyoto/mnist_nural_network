from network import network
from layer import layer_none
import numpy as np
from databaseHundler import dataHundler
def main():
    dh=dataHundler()
    images=dh.loadImgData("train-images-idx3-ubyte.gz")
    labels=dh.loadLabelData("train-labels-idx1-ubyte.gz")
    for i in range(10,20):
        dataHundler.showData(images[i],labels[i])


    #todo: add the mnist database
    network_test=network()
    img=np.random.randn(784)
    print(network_test.get_output(img=img))



if __name__ == "__main__":
    main()