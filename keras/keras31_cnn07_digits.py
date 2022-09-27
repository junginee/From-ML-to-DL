import numpy as np 
from sklearn.datasets import load_digits
from tensorflow.python.keras.models import Sequential,Model,load_model
from tensorflow.python.keras.layers import Dense,Input,Dropout,Flatten,Conv2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler

#1.데이터 
datasets = load_digits()
x= datasets.data
y= datasets.target

print(x.shape, y.shape) #(1797, 64) (1797, )
print(np.unique(y,return_counts=True)) #y의 라벨 ,[0 1 2 3 4 5 6 7 8 9]

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3,shuffle=True,
                                                     random_state=58)

print(x_train.shape,x_test.shape)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(1257,8,8,1)
x_test = x_test.reshape(540,8,8,1)
print(x_train.shape,x_test.shape)


#2.모델구성
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(2,2), padding='same', input_shape=(8,8,1))) 
model.add(Conv2D(7, (2,2), padding='same', activation='relu'))
model.add(Conv2D(7, (2,2), padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10,activation='softmax')) 



#3.컴파일,훈련
model.compile(loss= 'categorical_crossentropy', optimizer ='adam', metrics='accuracy') 

earlyStopping= EarlyStopping(monitor='val_loss',patience=30,mode='min',restore_best_weights=True,verbose=1)


model.fit(x_train, y_train, epochs=1000, batch_size=32,validation_split=0.2,callbacks=[earlyStopping], verbose=1)


#4.평가,예측
results = model.evaluate(x_test,y_test)
print('loss : ', results[0])

loss = model.evaluate(x_test,y_test)
print('loss: ', round(loss[0],4))

y_predict = model.predict(x_test) 
y_predict = y_predict.round()
acc = accuracy_score(y_test, y_predict) 
print('acc스코어: ', round(acc,4))

#================================= loss, accuracy ===================================#
# loss:  0.1465
# acc스코어:  0.9481
#=================================================================================#
