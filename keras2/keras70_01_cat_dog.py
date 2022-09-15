import numpy as np 
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping
from keras.models import Sequential,Model
from keras.layers import Dense,Flatten,GlobalAveragePooling2D,Input
from keras.applications import VGG19,VGG16,Xception,ResNet50,ResNet101,InceptionResNetV2,InceptionV3,DenseNet121,MobileNetV2,NASNetMobile,EfficientNetB0
from keras.datasets import cifar10,cifar100
import tensorflow as tf 
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
tf.random.set_seed(123)
import warnings
warnings.filterwarnings('ignore')

x_train = np.load('d:/study_data/_save/_npy/cat_dog_2/keras49_06_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/cat_dog_2/keras49_06_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/cat_dog_2/keras49_06_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/cat_dog_2/keras49_06_test_y.npy')

x_train = x_train.reshape(64, 150*150*3)
x_test = x_test.reshape(64, 150*150*3)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(64, 150,150,3)
x_test = x_test.reshape(64, 150,150,3)

#2.모델
# input1 = Input(shape=(150,150,3))
# vgg16 = VGG16(include_top=False)(input1)
# gap1 = Flatten()(vgg16)
# # gap1 = GlobalAveragePooling2D()(vgg16)    
# hidden1 = Dense(100)(gap1)
# output = Dense(1,activation='sigmoid')(hidden1)
# model = Model(inputs=input1,outputs=output)


# from sklearn.metrics import accuracy_score
# model.compile(loss= 'binary_crossentropy',optimizer='adam')
# model.fit(x_train,y_train, epochs=100, batch_size=256, verbose=1)
# model.evaluate(x_test,y_test)
# y_predict = np.argmax(model.predict(x_test),axis=1)
# acc = accuracy_score(y_test,y_predict)
# print('acc : ', round(acc,4))

m_list=[VGG19,VGG16,Xception,ResNet50,ResNet101,InceptionResNetV2,InceptionV3,DenseNet121,MobileNetV2,EfficientNetB0]

for i in m_list:

    vgg19 = i(weights='imagenet', include_top=False,
                input_shape=(150, 150, 3))  

    # vgg19.trainable= False     # vgg19 가중치 동결

    model = Sequential()
    model.add(vgg19)
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(1,activation='sigmoid'))
    model.layers[0].trainable= False
  
    # model.trainable = False

    from sklearn.metrics import accuracy_score
    model.compile(loss='binary_crossentropy',optimizer='adam')
    model.fit(x_train,y_train, epochs=100, batch_size=256, verbose=0)
    model.evaluate(x_test,y_test)
    y_predict = (model.predict(x_test)).round()
    acc = accuracy_score(y_test,y_predict)
    print(i.__name__,'acc : ',acc)

######### vgg16 #########
# all true -        acc :  0.4375
# vgg false -       acc :  0.7344
# all model false - acc :  0.6094



# VGG19 acc :  0.734375
# VGG16 acc :  0.734375
# Xception acc :  0.8125
# ResNet50 acc :  0.671875
# ResNet101 acc :  0.609375
# InceptionResNetV2 acc :  0.953125
# InceptionV3 acc :  0.78125
# DenseNet121 acc :  0.921875
# MobileNetV2 acc :  0.84375
# EfficientNetB0 acc :  0.640625