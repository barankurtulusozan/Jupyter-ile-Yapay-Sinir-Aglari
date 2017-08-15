import numpy as np

# X = (uyku, calisma), y = Sonuc
X = np.array(([3,5], [5,1], [10,2]), dtype=float)
y = np.array(([75], [82], [93]), dtype=float)

# Normalizasyon
X = X/np.amax(X, axis=0)
y = y/100 #Maximum 100