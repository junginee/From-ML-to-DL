import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Flatten,GlobalAveragePooling2D
from keras.applications import VGG19,inception_v3
from keras.datasets import cifar100
(x_train,y_train),(x_test,y_test) = cifar100.load_data()
print(x_train.shape,y_train.shape)  # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape,y_test.shape)    # (10000, 32, 32, 3) (10000, 1)


#1. 데이터
# model= vgg16.VGG16() # include_top = True,input_shape=(224,224,3)이 디폴트
incep_v3 = inception_v3(weights='imagenet',include_top=False,
                    input_shape=(32,32,3))

# VGG16.summary() # Trainable params: 14,714,688
# VGG16.trainable=False
# VGG16.summary() # Non-trainable params: 14,714,688
model = Sequential()
model.add(incep_v3)
model.add(GlobalAveragePooling2D())
model.add(Dense(100))
model.add(Dense(100,activation='softmax'))
# model.trainable =False
model.summary()

#3. 컴파일, 훈련
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=10,mode='auto')
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam')

model.fit(x_train,y_train,epochs=100,batch_size=4000,
          callbacks=[es],validation_split=0.3)
from sklearn.metrics import accuracy_score
#4. 평가, 예측
model.evaluate(x_test,y_test)
y_predcit = np.argmax(model.predict(x_test),axis=1)

acc = accuracy_score(y_test,y_predcit)

print("acc : ",acc)

################### include_top=False 로 진행
# acc :  0.7153
################### VGG16.trainable = False 로 진행
# acc :  0.579
################### model.trainable = False 로 진행
# acc :  0.1021
