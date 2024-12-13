from network import network
from layer import layer_none
import numpy as np
def main():
    #todo: add the mnist database
    network_test=network()
    img=np.random.randn(784)
    print(network_test.get_output(img=img))


if __name__ == "__main__":
    main()