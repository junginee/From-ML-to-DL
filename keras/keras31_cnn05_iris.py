#다중분류 point) --- loss categorical!, softmax ,마지막 노드갯수!,one hot encoding, argmax
import numpy as np 
from sklearn.datasets import load_iris
from tensorflow.python.keras.models import Sequential,Model,load_model
from tensorflow.python.keras.layers import Dense,Input,Dropout,Flatten,Conv2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler
import tensorflow as tf

#1.데이터
datasets = load_iris()

x= datasets['data']
y= datasets['target']

# print(x.shape,y.shape) #(150,4) (150, )


#################### one hot encoding ####################### 
print('y의 라벨값 : ', np.unique(y,return_counts=True)) 

#텐서플로우
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

print(y)
print(y.shape) #(150, 3)
############################################################### 

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3,shuffle=True,
                                                     random_state=58)


print(x_train.shape,x_test.shape) #(105, 4) (45, 4)


scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(105,2,2,1)
x_test = x_test.reshape(45,2,2,1)
print(x_train.shape,x_test.shape) #(105, 2, 2, 1) (45, 2, 2, 1)


#2.모델구성
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(2,2), padding='same', input_shape=(2,2,1))) 
model.add(Conv2D(7, (2,2), padding='same', activation='relu'))
model.add(Conv2D(7, (2,2), padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3,activation='softmax')) 


#3.컴파일,훈련
model.compile(loss= 'categorical_crossentropy', optimizer ='adam', metrics='accuracy') 


earlyStopping= EarlyStopping(monitor='val_loss',patience=30,mode='min',restore_best_weights=True,verbose=1)


model.fit(x_train, y_train, epochs=500, batch_size=12,validation_split=0.2,callbacks=[earlyStopping] ,verbose=1)

#4.평가,예측

results = model.evaluate(x_test,y_test)
print('loss : ', results[0])


y_predict = model.predict(x_test) # x값 4번째까지
y_predict = y_predict.argmax(axis=1) # 최대값의 위치 구해줌. argmin은 최솟값 (n, 3)에서(n, 1)로 변경됨.


y_test = y_test.argmax(axis=1) # y_test 값도 최대값 추출해줘야함 (n, 3) 에서 (n, 1)로 변경 
acc = accuracy_score(y_test,y_predict)# acc 정수값을 원한다. 
print('acc : ',acc)

#================================= loss, accuracy ===================================#
# loss :  0.09789939224720001
# acc :  0.9555555555555556
#=================================================================================#