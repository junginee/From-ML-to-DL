
#model1  model2
#   ㅁ    ㅁ
#     ＼ ／
#   (merge1)
#     ／ ＼
#   ㅁ     ㅁ
#output1  output2


#1. 데이터
import numpy as np
x1_datasets = np.array([range(100), range(301,401)]) #삼성전자 종가, 하이닉스 종가
x1 = np.transpose(x1_datasets)

print(x1.shape) #(100,2) (100,3) (100,2)

y1 = np.array(range(2001,2101)) #금리 
y2 = np.array(range(201,301)) #환율

from sklearn.model_selection import train_test_split
x1_train, x1_test,y1_train, y1_test ,y2_train, y2_test= train_test_split(
    x1, y1,y2, random_state = 66, test_size = 0.3)



print(x1_train.shape,x1_test.shape)   #(70, 2) (30, 2)
print(y1_train.shape, y1_test.shape)    #(70,) (30,)
print(y2_train.shape, y2_test.shape)    #(70,) (30,)



#2.모델구성 (#Sequential model은 복잡한 모델 만드는데 한계가 있으므로 함수형 모델 사용)
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input

#------------------------------X모델 구성------------------------------

input1 = Input(shape=(2,))
dense1 = Dense(10, activation = 'relu', name = 'ys1')(input1)
dense2 = Dense(20, activation = 'relu', name = 'ys2')(dense1)
dense3 = Dense(30, activation = 'relu', name = 'ys3')(dense2)
output1 = Dense(10, activation = 'relu', name = 'out_ys1')(dense3)

#------------------------------y모델 구성------------------------------

#Concatenate 
from tensorflow.python.keras.layers import concatenate #concatenate #(함수)     
#Concatenate : C가 대문자일 경우 class에 해당

merge = concatenate([output1], name = 'mg')

merge1 = Dense(20, activation = 'relu', name = 'mg1')(merge)
merge2 = Dense(30, name = 'mg2')(merge1)
merge3 = Dense(30, name = 'mg3')(merge2)
last_output = Dense(1, name='last1')(merge3)

#output 모델1
output41 = Dense(10)(last_output)
output42 = Dense(10)(output41)
last_output2 = Dense(1)(output42)


#output 모델2
output51 = Dense(10)(last_output)
output52 = Dense(10)(output51)
output53 = Dense(10)(output52)
last_output3 = Dense(1)(output53)

model = Model(inputs=[input1], outputs = [last_output2,last_output3])

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer = 'adam')
              #metrics = ['mse'])

model.fit([x1_train] ,[y1_train,y2_train] ,epochs=300, batch_size=16,
                 #validation_split=0.2,             
                 verbose=1)


#4. 평가, 예측
loss= model.evaluate([x1_test] ,[y1_test, y2_test])
print('loss(y) : ', loss)


from sklearn.metrics import r2_score
y1_predict, y2_predict = model.predict([x1_test])  
r2_1 = r2_score(y1_test, y1_predict)
r2_2 = r2_score(y2_test, y2_predict)
print('y1의 r2 스코어: ', round(r2_1,3))  
print('y2의 r2 스코어: ', round(r2_2,3))  

# loss(y) :  [706.4987182617188, 11.214268684387207, 695.284423828125]
#             ㄴ y1 + y2의 loss  / ㄴ y1의 loss  /  ㄴ y2의 loss
# y1의 r2 스코어:  0.987
# y2의 r2 스코어:  0.205

