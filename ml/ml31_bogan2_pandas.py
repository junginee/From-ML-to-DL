from statistics import median
import numpy as np
import pandas as pd

data = pd.DataFrame([ [2, np.nan, 6, 8, 10 ],
                      [2, 4, np.nan, 8, np.nan],
                      [2, 4, 6, 8, 10],
                      [np.nan, 4, np.nan, 8, np.nan]
                    ])

print(data.shape)       # (4,5)
data = data.transpose()
data.columns = ['x1', 'x2', 'x3', 'x4']
print(data)
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

print(data.isnull())
#       x1     x2     x3     x4
# 0  False  False  False   True
# 1   True  False  False  False
# 2  False   True  False   True
# 3  False  False  False  False
# 4  False   True  False   True

print(data.isnull().sum())
# x1    1          
# x2    2
# x3    0
# x4    3
# dtype: int64

print(data.info())
# Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   x1      4 non-null      float64
#  1   x2      3 non-null      float64
#  2   x3      5 non-null      float64
#  3   x4      2 non-null      float64

#1. 결측치 삭제
print("=============결측치 삭제===============")
print(data.dropna())  # axis default = 0 
print(data.dropna(axis=1))

#     x1   x2   x3   x4
# 3  8.0  8.0  8.0  8.0
#      x3
# 0   2.0
# 1   4.0
# 2   6.0
# 3   8.0
# 4  10.0

# 2-1. 특정값 - 평균
print("=============결측치 처리 mean()===============")
means = data.mean()   #컬럼별 평균
print("평균 : ", means)
data2 = data.fillna(means)
print(data2)

# 2-2. 특정값 - 중위값
print("=============결측치 처리 median()===============")
median = data.median()   #컬럼별 평균
print("평균 : ", median)
data3 = data.fillna(median)
print(data3)

# 2-3. 특정값 - ffill, bfill
print("=============결측치 처리 ffill, bfill===============")
data4 = data.fillna(method = 'ffill')
print(data4)

data5 = data.fillna(method='bfill')
print(data5)

#2-4. 특정값 - 임의값으로 채우기
print("=============결측치 - 임의값으로 채우기===============")
data6 = data.fillna(value = 77777)
print(data6)

###################### 특정컬럼만 !! ###########################
print("=============특정 컬럼만!!!===============")

means = data['x1'].mean()
print(means)
data['x1'] = data['x1'].fillna(means)
print(data)

meds = data['x2'].median()
print(meds)
data['x2'] = data['x2'].fillna(meds)
print(data)

data['x4'] = data['x4'].fillna(77777)
print(data)
