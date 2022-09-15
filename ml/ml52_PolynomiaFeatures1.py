import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

x = np.arange(12).reshape(4,3)

print(x)
# [[0 1]
#  [2 3]
#  [4 5]
#  [6 7]]
print(x.shape) #(4, 2)

pf = PolynomialFeatures(degree=2) #degree=3 넘을 경우 성능 떨어짐
x_pf = pf.fit_transform(x)

print(x_pf)
# [[ 1.  0.  1.  0.  0.  1.]       
#  [ 1.  2.  3.  4.  6.  9.]       
#  [ 1.  4.  5. 16. 20. 25.]       
#  [ 1.  6.  7. 36. 42. 49.]] 

print(x_pf.shape) #(4, 6)
