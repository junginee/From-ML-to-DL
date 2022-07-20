

#1. 데이터
import numpy as np
x1_datasets = np.array([range(100), range(301,401)]) #삼성전자 종가, 하이닉스 종가
x2_datasets = np. array([range(101,201), range(411,511), range(150,250)]) #원유, 돈육, 일
x1 = np.transpose(x1_datasets)
x2 = np.transpose(x2_datasets)

print(x1.shape, x2.shape) #(100,2) (100,3)

y = np.array(range(2001,2101)) #금리 

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(
    x1, x2, y, random_state = 66, test_size = 0.3)

print(x1_train.shape,x1_test.shape)   #(70, 2) (30, 2)
print(x2_train.shape, x2_test.shape)  #(70, 3) (30, 3)
print(y_train.shape, y_test.shape)    #(70,) (30,)



#2.모델구성 (#Sequential model은 복잡한 모델 만드는데 한계가 있으므로 함수형 모델 사용)
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input

#2-1. 모델1

input1 = Input(shape=(2,))
dense1 = Dense(1, activation = 'relu', name = 'ys1')(input1)
dense2 = Dense(2, activation = 'relu', name = 'ys2')(dense1)
dense3 = Dense(3, activation = 'relu', name = 'ys3')(dense2)
output1 = Dense(10, activation = 'relu', name = 'out_ys4')(dense3)

#2-2. 모델2

input2 = Input(shape=(3,))
dense11 = Dense(11, activation = 'relu', name = 'ys11')(input2)
dense12 = Dense(12, activation = 'relu', name = 'ys12')(dense11)
dense13 = Dense(13, activation = 'relu', name = 'ys13')(dense12)
dense14 = Dense(14, activation = 'relu', name = 'ys14')(dense13)
output2 = Dense(10, activation = 'relu', name = 'out_ys2')(dense14)

from tensorflow.python.keras.layers import concatenate, Concatenate
merge1 = concatenate([output1, output2], name = 'mg1')
merge2 = Dense(2, activation = 'relu', name = 'mg2')(merge1)
merge3 = Dense(3, name = 'mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

model = Model(inputs=[input1,input2], outputs = last_output)
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer = 'adam', metrics = ['mse'])

model.fit([x1_train,x2_train] ,y_train ,epochs=1, batch_size=32,
                 validation_split=0.2,             
                 verbose=1)


#4. 평가, 예측
loss= model.evaluate([x1_test,x2_test] ,y_test)
print('loss : ', loss)


from sklearn.metrics import r2_score
y_predict = model.predict([x1_test,x2_test])  
r2 = r2_score(y_test, y_predict)
print('r2 스코어: ', r2)  