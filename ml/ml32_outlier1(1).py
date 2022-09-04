import numpy as np
aaa = np.array([1,2,-10,4,5,6,7,8,50,10])

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

outliers_loc = outliers(aaa)
print("이상치의 위치 :", outliers_loc)    

import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show()

#이상치의 위치 : (array([2, 8], dtype=int64),)
# -10 과 50이 이상치

################################################
# 1사분위 :  2.5
# q2 :  5.5
# 3사분위 :  7.75
# iqr :  5.25
#################################################

# aaa = np.array([1,2,-10,4,5,6,7,8,50,10])
# [-10, 1, 2, 4, 5, 6, 7, 8, 10, 50]

# Q1 : (총도수-1) *0.25 + 1의 위치값
#      (10-1) * 0.25 +1 = 3.25의 위치값 = 2 + (4-2)*0.25 = 2.5
# Q2 : (5+6) / 2 = 5.5
# Q3 : (총도수-1) *0.75 +1의 위치값
#       (10-1) * 0.75 +1의 위치값 7.75의 위치값
#       7 + (8-7) *0.75 = 7.75
# IQR = 7.75 - 2.5 =  5.25
#                                                -> (-7.875)
# lower_bound = quartile_1 - (iqr * 1.5) = 2.5 - (5.25 * 1.5) = -5.375
# upper_bound = quartile_3 + (iqr * 1.5) = 7.75 + (5.25 * 1.5) = 15.625
# -5.375 이하의 값 or 15.625 이상의 값이 있는 데이터의 위치를 반환해라                            

#이상치의 위치 : (array([2, 8], dtype=int64),) =>> 이상치 : -10 ,50

# https://colinch4.github.io/2020-12-04/outlier/
