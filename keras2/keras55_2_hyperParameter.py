import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dropout, Dense, Conv2D, Flatten, MaxPool2D, Input

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(10000,28*28).astype('float32')/255.

#2. 모델
def build_model(drop=0.5, optimizer='adam', activation='relu', 
                            node1=64, node2=64, node3=64, lr=0.001):
    inputs = Input(shape=(28*28), name='input')
    x = Dense(node1, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node3, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax' ,name='outputs')(x)
    
    model = Model (inputs = inputs, outputs=outputs)
    
    model.compile(optimizer=optimizer, metrics=['acc'], 
                            loss='sparse_categorical_crossentropy')
    return model
    
def create_hyperparameter(): #255번 훈련
        batchs = [100, 200, 300, 400, 500]
        optimizers = ['adam', 'rmsprop', 'adadelta']
        dropout = [0.3, 0.4, 0.5]
        activation = ['relu', 'linear', 'sigmoid', 'selu', 'elu']  
        return{'batch_size' : batchs, 'optimizer' : optimizers, 
                    'drop' : dropout, 'activation' : activation}     #딕셔너리 형태 {}

hyperparameters  = create_hyperparameter()
print(hyperparameters)
# {'batch_size': [100, 200, 300, 400, 500], 'optimizer': ['adam', 'rmsprop', 'adadelta'], 
#  'drop': [0.3, 0.4, 0.5], 'activation': ['relu', 'linear', 'sigmoid', 'selu', 'elu']}    

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
keras_model = KerasClassifier(build_fn=build_model, verbose=1) 

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
model = RandomizedSearchCV(keras_model, hyperparameters, cv=2, n_iter=2, verbose=1) #총 15번 훈련

import time
start = time.time()
model.fit(x_train, y_train, epochs=7, validation_split=0.4) 
end = time.time() - start

print('걸린시간 : ', end - start)
print('best_params: ', model.best_params_)
print('model.best_setimator_ : ',model.best_estimator_)
print('best_score : ',model.best_score_)
print('score : ',model.score)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
print('accuracy_score : ', accuracy_score(y_test, y_predict))

#accuracy_score :  0.9493