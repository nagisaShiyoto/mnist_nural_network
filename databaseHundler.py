import gzip
import numpy as np
from matplotlib import pyplot as plt


"""
0.T-shirt(top)
1.pants(Trouser)
2.footer(Pullover)
3.Dress
4.Coat
5.Sandal
6.long sleave Shirt(shirt)
7.shoe(sneaker)
8.Bag
9.Ankle boot
"""

#sdding tests and documantation:)
class dataHundler:
    def __init__(self,path="C:\\Users\\eylon\\Desktop\\projects\\mnist_nural_network\\database"):
        self.path=path
    
    def loadImgData(self,fileName):
        path=self.path+"\\"+fileName

        with gzip.open(path) as file:
            magicNumber = int.from_bytes(file.read(4),"big")#start with magic number
            #gets the data about the pics
            n_images = int.from_bytes(file.read(4), 'big')
            n_rows = int.from_bytes(file.read(4), 'big')
            n_cols = int.from_bytes(file.read(4), 'big')

            buffer=file.read(n_images*n_rows*n_cols)
            allData=np.frombuffer(buffer,dtype=np.uint8)
            allData=allData.reshape(n_images,n_rows,n_cols)
        return allData
    
    
    def loadLabelData(self,fileName):
        path=self.path+"\\"+fileName

        with gzip.open(path) as file:
            magic_number = int.from_bytes(file.read(4), 'big')
            n_labels = int.from_bytes(file.read(4), 'big')

            buffer = file.read(n_labels)
            allData=np.frombuffer(buffer,dtype=np.uint8)

        return allData
    

    def showData(img,label):
        
        plt.imshow(img, cmap='gray')
        plt.xlabel(label)
        plt.show()

    def loadTraining(self):
        self.trainingImg=self.loadImgData("train-images-idx3-ubyte.gz")
        self.trainingLabels=self.loadLabelData("train-labels-idx1-ubyte.gz")
    
    def loadTest(self):
        self.testingImg = self.loadImgData("t10k-images-idx3-ubyte.gz")
        self.testingLabels = self.loadLabelData("t10k-labels-idx1-ubyte.gz")
    
    def getTrainingData(self):
        return self.testingImg,self.trainingLabels

    def getTest(self):
        return self.testingImg,self.testingLabels