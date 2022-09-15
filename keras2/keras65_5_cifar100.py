import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Flatten,GlobalAveragePooling2D
from keras.applications import vgg16,resnet
from keras.datasets import cifar100
(x_train,y_train),(x_test,y_test) = cifar100.load_data()
from tensorflow.keras.utils import to_categorical
print(x_train.shape,y_train.shape)  # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape,y_test.shape)    # (10000, 32, 32, 3) (10000, 1)

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
print(y_train.shape,y_test.shape) # (50000, 100) (10000, 100)

#1. 데이터
# model= vgg16.VGG16() # include_top = True,input_shape=(224,224,3)이 디폴트
# VGG16 = vgg16.VGG16(weights='imagenet',include_top=False,
#                     input_shape=(32,32,3))
RESNET = resnet.ResNet50(weights='imagenet',include_top=False,
                       input_shape=(32,32,3))
# VGG16.summary() # Trainable params: 14,714,688
# RESNET.trainable=False
# VGG16.summary() # Non-trainable params: 14,714,688
model = Sequential()
model.add(RESNET)
model.add(GlobalAveragePooling2D())
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='sigmoid'))
model.add(Dense(128, activation='relu'))
model.add(Dense(100,activation='softmax'))
# model.trainable =False
model.summary()

#3. 컴파일, 훈련
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=15,mode='auto')
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam')

model.fit(x_train,y_train,epochs=500,batch_size=3000,
          callbacks=[es],validation_split=0.3)
from sklearn.metrics import accuracy_score
#4. 평가, 예측
model.evaluate(x_test,y_test)
y_predcit = np.argmax(model.predict(x_test),axis=1)
# y_predcit = to_categorical(y_predcit)

acc = accuracy_score(y_test,y_predcit)

print("acc : ",acc)
################### include_top=False 로 진행
# acc :  0.4791
################### VGG16.trainable = False 로 진행
# acc :  0.3228
################### model.trainable = False 로 진행
# acc :  0.0124