import numpy as np 
from sklearn.datasets import load_wine
from tensorflow.python.keras.models import Sequential,Model,load_model
from tensorflow.python.keras.layers import Dense,Input,Dropout,LSTM, Flatten, Conv1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler

#1.데이터 
datasets = load_wine()
x= datasets.data
y= datasets.target

# print(x.shape, y.shape) #(178, 13) (178, )
# print(np.unique(y,return_counts=True)) #y의 라벨 ,[0 1 2]

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3,shuffle=True,
                                                     random_state=58)

print(x_train.shape,x_test.shape)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(124,13,1)
x_test = x_test.reshape(54,13,1)
print(x_train.shape,x_test.shape) #(124, 13, 1) (54, 13, 1)

#2.모델구성
model = Sequential()
model.add(Conv1D(64, 2, input_shape=(13,1))) 
model.add(Flatten())
model.add(Dense(7, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3,activation='softmax')) 


#3.컴파일,훈련
model.compile(loss= 'categorical_crossentropy', optimizer ='adam', metrics='accuracy') 
 
earlyStopping= EarlyStopping(monitor='val_loss',patience=30,mode='min',restore_best_weights=True,verbose=1)

model.fit(x_train, y_train, epochs=500, batch_size=16,validation_split=0.2,callbacks=[earlyStopping], verbose=1)


#4.평가,예측

results = model.evaluate(x_test,y_test)
print('loss : ', results[0])

loss = model.evaluate(x_test,y_test)
print('loss: ', round(loss[0],4))

y_predict = model.predict(x_test) 
y_predict = y_predict.round()
acc = accuracy_score(y_test, y_predict) 
print('acc스코어: ', round(acc,4))


#================================= [CNN]loss, accuracy ===========================#
# loss:  0.1957
# acc스코어:  0.9259
#=================================================================================#

#================================= [LSTM]loss, accuracy ==========================#
# loss:  0.2986
# acc스코어:  0.9074
#=================================================================================#

#================================= [Conv1D]loss, accuracy ==========================#
# loss:  0.4776
# acc스코어:  0.9074
#=================================================================================#