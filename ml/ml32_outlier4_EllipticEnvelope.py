import numpy as np
import pandas as pd
aaa = np.array([[-10,2,3,4,5,6,7,8,9,10,11,12,50],
               [100,200,-30,400,500,600,-70000,800,900,1000,110,420,350]])
aaa = np.transpose(aaa)
print(aaa.shape) 


from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination=.5)
pred = outliers.fit_predict(aaa)
print(pred.shape) # (13,)

b = list(pred)
print(b.count(-1))
index_for_outlier = np.where(pred == -1)
print('outier indexex are', index_for_outlier)

outlier_value = aaa[index_for_outlier]
print('outlier_value :', outlier_value) 