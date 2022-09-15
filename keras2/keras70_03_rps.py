import numpy as np 
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping
from keras.models import Sequential,Model
from keras.layers import Dense,Flatten,GlobalAveragePooling2D,Input
from keras.applications import VGG19,VGG16,Xception,ResNet50,ResNet101,InceptionResNetV2,InceptionV3,DenseNet121,MobileNetV2,EfficientNetB0
from keras.datasets import cifar10,cifar100
import tensorflow as tf 
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
tf.random.set_seed(123)
import warnings
warnings.filterwarnings('ignore')


x = np.load('d:/study_data/_save/_npy/rps_02/keras49_08_x.npy')
y = np.load('d:/study_data/_save/_npy/rps_02/keras49_08_y.npy')

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y, train_size=0.8,shuffle=True,random_state=66,stratify=y)

x_train = x_train.reshape(51, 150*150*3)
x_test = x_test.reshape(13, 150*150*3)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(51, 150,150,3)
x_test = x_test.reshape(13, 150,150,3)

# from keras.utils import to_categorical
# y_trian = to_categorical(y_train)
# y_test = to_categorical(y_test)
m_list=[VGG19,VGG16,Xception,ResNet50,ResNet101,InceptionResNetV2,InceptionV3,DenseNet121,MobileNetV2,EfficientNetB0]
acc_list=[]

for i in m_list:
    

    vGG16 = i(weights='imagenet', include_top=False,
                input_shape=(150, 150, 3))  

    # vGG16.trainable= False     

    model = Sequential()
    model.add(vGG16)
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(3,activation='softmax'))
    model.layers[0].trainable= False

    # model.trainable = False

    from sklearn.metrics import accuracy_score
    model.compile(loss= 'categorical_crossentropy',optimizer='adam',metrics=['acc'])
    model.fit(x_train,y_train, epochs=100, batch_size=256, verbose=0)
    model.evaluate(x_test,y_test)
    y_predict = model.predict(x_test)
    y_predict = np.argmax(y_predict,axis=1) 
    y_test = np.argmax(y_test,axis=1)
    acc = accuracy_score(y_test,y_predict)
    print('acc : ', round(acc,4))
    from keras.utils import to_categorical
    y_trian = to_categorical(y_train)
    y_test = to_categorical(y_test)
    acc_list.append([i.__name__,acc])
    
print(acc_list)

# [['VGG19', 0.46153846153846156], ['VGG16', 0.38461538461538464], 
#  ['Xception', 0.6153846153846154], ['ResNet50', 0.3076923076923077], 
#  ['ResNet101', 0.46153846153846156], ['InceptionResNetV2', 0.6923076923076923], 
#  ['InceptionV3', 0.6923076923076923], ['DenseNet121', 0.7692307692307693], 
#  ['MobileNetV2', 0.38461538461538464], ['EfficientNetB0', 0.3076923076923077]]

# model trainable False

# [['VGG19', 0.7692307692307693], ['VGG16', 0.5384615384615384], 
#  ['Xception', 0.5384615384615384], ['ResNet50', 0.9230769230769231],
#  ['ResNet101', 0.9230769230769231], ['InceptionResNetV2', 0.6153846153846154], 
# ['InceptionV3', 0.46153846153846156], ['DenseNet121', 0.8461538461538461], 
# ['MobileNetV2', 0.9230769230769231], ['EfficientNetB0', 0.9230769230769231]]