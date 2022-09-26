import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score, accuracy_score
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

# One Hot Encoding 
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)
print(y.shape)  # (150, 3)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=66
)

#2. 모델 구성
model = Sequential()
model.add(Dense(100, activation='linear', input_dim=4))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(100, activation='relu'))
model.add(Dense(3, activation='softmax'))   # 결과값 label이 3개이므로 output 노드의 갯수는 3이 됨 
                                            # softmax를 통해 제일 큰 값이 선택됨(softmax의 값은 전체 합계 1.0이 됨)
                                            # class: 1) Iris-Setosa  2) Iris-Versicolour  3) Iris-Virginica


#3. 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',    # 다중분류에서 loss = 'categorical_crossentropy'를 사용함
              metrics=['accuracy'])   
import datetime
date = datetime.datetime.now()      # 2022-07-07 17:21:42.275191
date = date.strftime("%m%d_%H%M")   # 0707_1723
print(date)

filepath = './_ModelCheckPoint/k26/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

earlyStopping = EarlyStopping(monitor = 'val_loss', patience=50, mode='min', verbose=1, 
                              restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
                      save_best_only=True, 
                      filepath="".join([filepath, '05_', date, '_', filename])
                      )
start_time = time.time()

hist = model.fit(x_train, y_train, epochs=1000, batch_size=10, 
                 validation_split=0.2,
                 callbacks=[earlyStopping, mcp],
                 verbose=1)
end_time = time.time() - start_time


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)       
print('accuracy : ', acc)



#================================= 1. 기본 출력 ===================================#
# loss :  0.07862314581871033
# accuracy :  0.9555555582046509
# 05_0707_1934_0203-0.0039.hdf5
#=================================================================================#

#================================ 2. dorpout 적용 ================================#
# loss :  0.0650220736861229
# accuracy :  0.9777777791023254
# 05_0708_1108_0101-0.0034.hdf5
#=================================================================================#
