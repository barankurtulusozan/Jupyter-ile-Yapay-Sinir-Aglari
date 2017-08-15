## ----------------------- Kısım 1 ---------------------------- ##
import numpy as np

# X = (uyku, calisma), y = Sonuc
X = np.array(([3,5], [5,1], [10,2]), dtype=float)
y = np.array(([75], [82], [93]), dtype=float)

# Normalizasyon
X = X/np.amax(X, axis=0)
y = y/100 #Maximum 100

## ----------------------- Kısım 2 ---------------------------- ##

class Neural_Network(object):
    def __init__(self):        
        #Hiperparametreler
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3
        
        #Ağırlıklar 
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)
        
    def forward(self, X):
        #Ağda giriş değerlerini ilerlet
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3) 
        return yHat
        
    def sigmoid(self, z):
        #Sigmoid fonksiyonunu skalare, vektör veya matrise uygulayabilirsin.
        return 1/(1+np.exp(-z))