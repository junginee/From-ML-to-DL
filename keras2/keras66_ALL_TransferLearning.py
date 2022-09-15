from keras.applications import VGG16, VGG19
from keras.applications import ResNet50, ResNet50V2
from keras.applications import ResNet101, ResNet101V2, ResNet152, ResNet152V2
from keras.applications import MobileNet, MobileNetV2
from keras.applications import DenseNet121, DenseNet169
from keras.applications import MobileNetV3Small, MobileNetV3Large
from keras.applications import NASNetLarge, NASNetMobile
from keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB7
from keras.applications import Xception
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()


models = [
                # VGG16, VGG19, 
                # ResNet50, ResNet50V2,
                # ResNet101, ResNet101V2, ResNet152, ResNet152V2,
                # MobileNet, MobileNetV2,  DenseNet121, DenseNet169,
                # MobileNetV3Small, MobileNetV3Large,
                NASNetLarge, NASNetMobile,
                EfficientNetB0, EfficientNetB1, EfficientNetB7,
                Xception
                ]

for model in models:
    model = model()
    model.trainable = False
    
    print("=========================")
    print("모델명 : ", model.name)
    print("전체 가중치 갯수 : ", len(model.weights))
    print("훈련 가능 가중치 갯수 : ", len(model.trainable_weights))
    
'''
=========================
모델명 :  vgg16
전체 가중치 갯수 :  32
훈련 가능 가중치 갯수 :  0
=========================
모델명 :  vgg19
전체 가중치 갯수 :  38
훈련 가능 가중치 갯수 :  0
=========================
모델명 :  resnet50
전체 가중치 갯수 :  320
훈련 가능 가중치 갯수 :  0
=========================
모델명 :  resnet50v2
전체 가중치 갯수 :  272
훈련 가능 가중치 갯수 :  0
=========================
모델명 :  resnet101
전체 가중치 갯수 :  626
훈련 가능 가중치 갯수 :  0
=========================
모델명 :  resnet101v2
전체 가중치 갯수 :  544
훈련 가능 가중치 갯수 :  0
=========================
모델명 :  resnet152
전체 가중치 갯수 :  932
훈련 가능 가중치 갯수 :  0
=========================
모델명 :  resnet152v2
전체 가중치 갯수 :  816
훈련 가능 가중치 갯수 :  0
=========================
모델명 :  mobilenet_1.00_224
전체 가중치 갯수 :  137
훈련 가능 가중치 갯수 :  0
=========================
모델명 :  mobilenetv2_1.00_224
전체 가중치 갯수 :  262
훈련 가능 가중치 갯수 :  0
=========================
모델명 :  densenet121
전체 가중치 갯수 :  606
훈련 가능 가중치 갯수 :  0
=========================
모델명 :  densenet169
전체 가중치 갯수 :  846
훈련 가능 가중치 갯수 :  0
=========================
모델명 :  MobilenetV3small
전체 가중치 갯수 :  210
훈련 가능 가중치 갯수 :  0
=========================
모델명 :  MobilenetV3large
전체 가중치 갯수 :  266
훈련 가능 가중치 갯수 :  0
=========================
모델명 :  NASNet
전체 가중치 갯수 :  1546  
훈련 가능 가중치 갯수 :  0
=========================
모델명 :  NASNet
전체 가중치 갯수 :  1126  
훈련 가능 가중치 갯수 :  0
=========================
모델명 :  efficientnetb0  
전체 가중치 갯수 :  314   
훈련 가능 가중치 갯수 :  0
=========================
모델명 :  efficientnetb1  
전체 가중치 갯수 :  442   
훈련 가능 가중치 갯수 :  0
=========================
모델명 :  efficientnetb7
전체 가중치 갯수 :  1040
훈련 가능 가중치 갯수 :  0
=========================
모델명 :  xception
전체 가중치 갯수 :  236
훈련 가능 가중치 갯수 :  0
'''    