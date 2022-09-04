import numpy as np
aaa = np.array([-10,2,3,4,5,6,7,8,9,10,11,12,50])
aaa = aaa.reshape(1,-1)
from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination=.3) 
                                          #.숫자 = 이 범위의 데이터를 이상치로 설정   

outliers.fit(aaa)
results = outliers.predict(aaa)
print(results)
