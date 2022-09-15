import numpy as np
# from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Input,Dropout
from tensorflow.python.keras.layers import GlobalAveragePooling2D
from keras.datasets import mnist,cifar100

#1. 데이터
(x_train,y_train),(x_test,y_test) = mnist.load_data()

print(x_train.shape) # (60000, 28, 28)

x_train = x_train.reshape(60000,28*28)
x_test = x_test.reshape(10000,28*28)
from sklearn.preprocessing import MinMaxScaler,StandardScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델
dropout=0.5
activation='relu'
inputs = Input(shape=(28,28,1), name = 'input')
x = Conv2D(64,kernel_size=(2,2),padding='valid',
           activation=activation, name= 'hidden1')(inputs)
x = Dropout(dropout)(x) # 27 ,27, 128
x = Conv2D(32,kernel_size=(2,2),padding='same',
           activation=activation, name= 'hidden2')(x)
x = Dropout(dropout)(x) # 27 ,27, 64
x = MaxPool2D()(x)
x = Conv2D(32,kernel_size=(3,3),padding='valid',
           activation=activation, name= 'hidden3')(x)
x = Dropout(dropout)(x) # 25, 25, 32 
# x = Flatten()(x) # (25*25*32) # 20000 연산량이 급증하여 시간이 오래걸리며 오히려 과적합 위험있음
x = GlobalAveragePooling2D()(x)

x = Dense(100, activation=activation, name='hidden4')(x)
x = Dropout(dropout)(x)
outputs = Dense(10, activation='softmax', name='outputs')(x)

model = Model(inputs=inputs,outputs=outputs)
model.summary()

#3. 컴파일,  훈련
from tensorflow.python.keras.optimizers import adam_v2
learning_rate = 0.01
optimizer = adam_v2.Adam(lr=learning_rate)

model.compile(optimizer=optimizer,metrics=['acc'],
                loss='categorical_crossentropy')



# {'batch_size': [100, 200, 300, 400, 500],
#  'optimizer': ['adam', 'rmsprop', 'adadelta'], 
# 'dropout': [0.3, 0.4, 0.5],
# 'activation': ['relu', 'linear', 'sigmoid', 'selu', 'elu']}
from tensorflow.python.keras.callbacks import EarlyStopping,ReduceLROnPlateau
import time
es = EarlyStopping(monitor='val_loss',patience=20,mode='min',verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',patience=10,
                              mode='auto',verbose=1,factor=0.5)
start_time= time.time()
model.fit(x_train,y_train, epochs=100, 
          validation_split=0.2,batch_size=1000,
          callbacks=[es,reduce_lr])
end_time = time.time()
loss,acc = model.evaluate(x_test,y_test)

from sklearn.metrics import accuracy_score
# y_predict = np.argmax(model.predict(x_test),axis=1)
print('걸린 시간 : ',end_time-start_time)

print("loss :",loss)
print("acc :", acc)

# print('accuracy_score : ',accuracy_score(y_test,y_predict))

# Epoch 00098: 
# ReduceLROnPlateau 
# reducing learning rate to 0.0012499999720603228.
# Epoch 00098: early stopping
# 걸린 시간 :  316.158518075943
# loss : 0.12253511697053909
# acc : 0.9696999788284302