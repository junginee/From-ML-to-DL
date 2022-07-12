import numpy as np 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,Input,Dropout,Flatten,Conv2D
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint

#1.데이터
datasets = load_breast_cancer()

x = datasets['data']
y = datasets['target']

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size= 0.7,random_state=31)

print(x_train.shape,x_test.shape)


# # minmax , standard
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(398,5,3,2)
x_test = x_test.reshape(171,5,3,2)
print(x_train.shape,x_test.shape)



#2.모델구성
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(2,2), padding='same', input_shape=(5,3,2))) 
model.add(Conv2D(16, (2,2), padding='same', activation='relu'))
model.add(Conv2D(16, (2,2), padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1)) 

# import datetime 
# date = datetime.datetime.now()
# date = date.strftime('%m%d_%H%M')

earlyStopping= EarlyStopping(monitor= 'val_loss',patience=30,mode='min',restore_best_weights=True,verbose=1) 


#3.컴파일,훈련
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy','mse'],) 

# filepath ='./_ModelCheckpoint/k24/'
# filename ='{epoch:04d}-{val_loss:.4f}.hdf5'
                                                                             
# mcp = ModelCheckpoint(monitor = 'val_loss',mode= 'auto',verbose=1,save_best_only=True,
#                       filepath = "".join([filepath,'cancer',date,'_',filename]))

hist=model.fit(x_train,y_train,epochs=500, batch_size=16,verbose=1,validation_split=0.2, callbacks= [earlyStopping])


#4.평가,예측
loss = model.evaluate(x_test,y_test)
print('loss: ', round(loss[0],4))

y_predict = model.predict(x_test) 
y_predict = y_predict.round()
acc = accuracy_score(y_test, y_predict) 
print('acc스코어: ', round(acc,4))

#================================= loss, accuracy ===================================#
# loss:  0.0599
# acc스코어:  0.2749
#=================================================================================#