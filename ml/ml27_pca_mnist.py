import numpy as np
from sklearn.decomposition import PCA
from keras.datasets import mnist
(x_train, _), (x_test, _) = mnist.load_data() 
                                            
                                              
x = np.append(x_train, x_test, axis=0) 
print(x.shape)  #(70000, 28, 28)

x = x.reshape(70000, 784)   
#x=x.reshape(x.shape[0], x.shape[1]*x.shape[2])

pca = PCA(n_components=784)
x = pca.fit_transform(x)
print(x)
print(x.shape) 

pca_EVR = pca.explained_variance_ratio_
print(pca_EVR)

cumsum=np.cumsum(pca_EVR)
print(cumsum)

print(np.argmax(cumsum >= 0.95) +1 ) #154
print(np.argmax(cumsum >= 0.99) +1 ) #331
print(np.argmax(cumsum >= 0.999) +1 ) #486
print(np.argmax(cumsum >= 1.0) +1)  # 713                                


import matplotlib.pyplot as plt
plt.plot(cumsum)
#plt.plot(pca_EVR)
plt.grid()
plt.show()
