import layer
import numpy as np
def main():
    lyers= layer.layer_none()
    
    a=np.array([[1,2,3],[1,2,3]])
    b=np.array([1,10,100])
    print(b.dot(a))

if __name__ == "__main__":
    main()