# 결측치 처리
#1. 행 또는 열 삭제
#2. 임의의 값
#   ㄴ 평균 mean
#   ㄴ 중위 median
#   ㄴ 0 (예 : 1 N 3 4 5 6 ) N에 0 넣어줌 by fillna
#   ㄴ 앞의 값 (예 : 1 N 3 4 5 6 ) N에 1 넣어줌 by ffill (f=front)
#   ㄴ 뒤의 값 (예 : 1 N 3 4 5 6 ) N에 3 넣어줌 by bfill (b=back)
#   ㄴ 특정 값, 기타 등등..
#3. 보간 - interpolate (선형회귀 방식)
#   ㄴ 빈 자리를 선형회귀 방식으로 찾아낸다.
#4. 모델 - predict (예측 값을 결측치 자리에 넣어준다.) 
#   ㄴhow? 결측치 모두 제거 후 모델 구성 및 훈련
        #  결측치값을 predict 값에 넣어 결측치에 대한 예측값을 뽑아내
        #  이 예측값을 결측치의 값에 넣어준다.
#5. 부스팅 계열 - 통상 결측치, 이상치에 대해 자유롭다.         

import pandas as pd
import numpy as np
from datetime import datetime

dates = ['8/10/2022','8/11/2022','8/12/2022','8/13,2022', '8/14/2022']

# dates = pd.to_datetime(dates)
# print(dates)

print("===========================")
ts = pd.Series([2, np.nan, np.nan, 8, 10], index=dates) 
print(ts)

print("===========================")
ts = ts.interpolate()
print(ts)