from keras.datasets import imdb
import numpy as np

(x_train, y_train),(x_test, y_test) = imdb.load_data(num_words= 1000)

print(x_train.shape, x_test.shape) #(25000,) (25000,)
print(len(np.unique(y_train))) #2

print(type(x_train), type(y_train)) #<class 'numpy.ndarray'> <class 'numpy.ndarray'>
print(type(x_train[0])) #<class 'list'> => pad_sequences를 통해 리스트를 넘파이 배열로 바꿔준다.

print(len(x_train[0])) #218
print(len(x_train[1])) #189

print('뉴스기사의 최대 길이 : ',max(len(i) for i in x_train)) #2494
print('뉴스기사의 평균일이 : ', sum(map(len, x_train)) / len(x_train)) #238.71364


#전처리
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

x_train = pad_sequences(x_train, padding = 'pre', maxlen=100, truncating = 'pre')
                        #앞에 0으로 채워주고 길이는 100이 넘는다면 앞 부분을 자른다.
x_test= pad_sequences(x_test, padding = 'pre', maxlen=100, truncating = 'pre')     

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, y_train.shape)  #(25000, 100) (25000, 2)     
print(x_test.shape, y_test.shape)  #(25000, 100) (25000, 2)


#2. 모델구성
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Embedding, Flatten

model = Sequential()                        
             
model.add(Embedding(input_dim=1000, output_dim=13, input_length=100)) #input_length 명시 or 생략 둘다 가능
model.add(Flatten())
model.add(Dense(16, activation='relu'))  
model.add(Dense(10, activation='relu'))  
model.add(Dense(5))  
model.add(Dense(2, activation='sigmoid'))  

model.summary()                                  

#3.컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=32)


#4.평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ' ,round(loss[0],4))
print('accuracy : ', round(loss[1],4))     
