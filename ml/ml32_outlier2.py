#다차원 이상치 출력

import numpy as np
import pandas as pd
aaa = np.array([[-10,2,3,4,5,6,7,8,9,10,11,12,50],
               [100,200,-30,400,500,600,-70000,800,900,1000,110,420,350]])
aaa = np.transpose(aaa)
print(aaa.shape) #(13, 2)


def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out,
                                               [25,50,75]) 
                                               # 하위 25% 위치 값 Q1
                                               # 하위 50% 위치 값 Q2 (중앙값)
                                               # 하위 75% 위치 값 Q3
    print("1사분위 : ", quartile_1)
    print("q2 : ", q2)
    print("3사분위 : ", quartile_3)
    iqr = quartile_3 -quartile_1
    print("iqr : ", iqr) 
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out>upper_bound) |
                    (data_out<lower_bound))


outliers_loc1 = outliers(aaa[:,0]) #모든행과 첫번째열
print("이상치의 위치1 :", outliers_loc1)  
# 이상치의 위치1 : (array([ 0, 12], dtype=int64),)   
print("\n")

outliers_loc2 = outliers(aaa[:,1]) #모든행과 두번째열
print("이상치의 위치2 :", outliers_loc2)  
# 이상치의 위치2 : (array([6], dtype=int64),)

##################################
for i in range(aaa.shape[1]):
    w = aaa[:, i]
    outliers_loc = outliers(w)
    print(i,'열의 이상치의 위치 :', outliers_loc,'\n')



#################################
# 1사분위 :  4.0
# q2 :  7.0
# 3사분위 :  10.0
# iqr :  6.0
# 이상치의 위치1 : (array([ 0, 12], dtype=int64),)   -10, 50

# 1사분위 :  110.0
# q2 :  400.0
# 3사분위 :  600.0
# iqr :  490.0
# 이상치의 위치2 : (array([6], dtype=int64),) -7,000
