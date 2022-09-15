from pickletools import optimize
from keras.models import Model
from keras.layers import Dense,Flatten,Input,GlobalAveragePooling2D
from keras.applications import VGG16, InceptionV3
from keras.datasets import cifar100
from sklearn.metrics import accuracy_score
import numpy as  np


(x_train,y_train),(x_test,y_test) = cifar100.load_data()
# print(x_train.shape,y_train.shape)  # (50000, 32, 32, 3) (50000, 1)
# print(x_test.shape,y_test.shape)    # (10000, 32, 32, 3) (10000, 1)
x_train = x_train.reshape(500,320,320,3) 
x_test = x_test.reshape(100,320,320,3)
y_train = y_train.reshape(500,100)
y_test = y_test.reshape(100,100)
# base_model.summary() # Total params: 21,802,784
input = Input(shape=(320,320,3))
# x = base_model.output # base_model 에 끝을 x로 정의
x = InceptionV3(weights='imagenet', include_top=False)(input) 
# 모델 자체가 요구하는 최소치 인풋쉐이프가 있어 cifar100은 훈련 불가
x = GlobalAveragePooling2D()(x)
x = Dense(102,activation='relu')(x)

output1 = Dense(100,activation='softmax')(x)

model = Model(inputs=input, outputs=output1)

#1.
# for layer in base_model.layers:
#     layer.trainable = False
    # Total params: 21,802,784
    # Trainable params: 0
    # Non-trainable params: 21,802,784

#2.
# base_model.trainable = False
# base_model.summary()
# Total params: 21,802,784
# Trainable params: 0
# Non-trainable params: 21,802,784

# print(len(base_model.layers)) # 311

#3. 컴파일, 훈련

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam')

model.fit(x_train,y_train,epochs = 10, batch_size = 2000)

#4. 평가, 예측

model.evaluate(x_test,y_test)

y_predict = model.predict(x_test)

print(y_predict)