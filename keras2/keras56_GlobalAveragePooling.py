from gc import callbacks
import numpy as np
from tensorflow.keras.datasets import mnist,cifar100
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Input,Dropout
import keras
import tensorflow as tf 
from tensorflow.keras.layers import GlobalAveragePooling2D
from sklearn.metrics import accuracy_score

# 1.데이터
(x_train,y_train),(x_test,y_test)  = cifar100.load_data()

x_train = x_train.reshape(50000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(50000, 32,32,3)
x_test = x_test.reshape(10000, 32,32,3)

from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# 2.모델 
activation = 'relu'
drop = 0.2
optimizer = 'adam'

inputs = Input(shape= (32, 32, 3), name= 'input')
x = Conv2D(64, (2, 2), padding='valid', activation=activation, name= 'hidden1')(inputs)
x = Dropout(drop)(x)
# x = Conv2D(64, (2, 2), padding='same', activation=activation, name= 'hidden2')(x)
# x = Dropout(drop)(x)
x = MaxPool2D()(x)
x = Conv2D(32, (3, 3), padding='valid', activation=activation, name= 'hidden3')(x)
x = Dropout(drop)(x)
x = GlobalAveragePooling2D()(x) 
x = Dense(100, activation=activation, name='hidden4')(x)
x = Dropout(drop)(x)
outputs = Dense(100, activation='softmax', name='outputs')(x)
model = Model(inputs=inputs, outputs=outputs)


from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', restore_best_weights=True,verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto', verbose=1, factor=0.5) #factor50프로 감축시키겟다

model.compile(optimizer=optimizer,metrics=['acc'], loss='categorical_crossentropy')
model.fit(x_train,y_train, epochs=100,callbacks=[es,reduce_lr], validation_split=0.4,batch_size=128) 


loss, acc = model.evaluate(x_test,y_test)
y_predic = model.predict(x_test)
# y_predict = np.argmax(model.predict(x_test),axis=-1)

print('loss : ', loss)
print('acc : ', acc)
# print('accuracy : ', accuracy_score(y_test,y_predict))