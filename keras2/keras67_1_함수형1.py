from keras.models import Model
from keras.layers import Dense,Flatten,Input
from keras.applications import VGG16
from keras.datasets import cifar100
from sklearn.metrics import accuracy_score
import numpy as  np


#1. 데이터
(x_train,y_train),(x_test,y_test) = cifar100.load_data()

#2. 모델
input = Input(shape=(32,32,3),name='input')
x = VGG16(weights='imagenet',include_top=False)(input)
x = Flatten()(x)
x = Dense(100,name='fc1')(x)
output = Dense(100,activation='softmax',name='output')(x)

model = Model(inputs=input, outputs=output)
model.summary()

#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam')

model.fit(x_train,y_train,epochs=10,batch_size=2000,verbose=1)

#4. 평가, 예측

model.evaluate(x_test,y_test)

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict,axis=1)

acc = accuracy_score(y_test,y_predict)

print('acc : ',acc)
# Model: "model"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  input (InputLayer)          [(None, 32, 32, 3)]       0

#  vgg16 (Functional)          (None, None, None, 512)   14714688

#  flatten (Flatten)           (None, 512)               0

#  fc1 (Dense)                 (None, 100)               51300

#  output (Dense)              (None, 100)               10100

# =================================================================
# Total params: 14,776,088
# Trainable params: 14,776,088
# Non-trainable params: 0
# _________________________________________________________________



# acc :  0.0109