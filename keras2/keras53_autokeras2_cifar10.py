import autokeras as ak
print(ak.__version__)       #1.0.20
import tensorflow as tf 
import keras
import time 

# 1.데이터
(x_train,y_train), (x_test, y_test) = \
    keras.datasets.cifar10.load_data()
    
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# 2.모델 
model = ak.ImageClassifier(
    overwrite= True,
    max_trials= 2,
)

# 3.컴파일,훈련
start = time.time()
model.fit(x_train, y_train, epochs=5)
end = time.time()

# 4.평가,예측 
y_predict = model.predict(x_test)

results = model.evaluate(x_test,y_test)
print('결과 : ', results)
print('걸린시간 : ', round(end - start, 4))

# 결과 :  [0.8699784874916077, 0.7016000151634216]
# 걸린시간 :  2704.0955