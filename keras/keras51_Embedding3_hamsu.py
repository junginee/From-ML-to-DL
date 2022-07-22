
from keras.preprocessing.text import Tokenizer
import numpy as np

#1. 데이터
docs = ['너무 재밌어요', '참 최고에요', '참 잘 만든 영화예요',
        '추천하고 싶은 영화입니다.', '한 번 더 보고 싶네요', '글세요',
        '별로에요', '생각보다 지루해요', '연기가 어색해요', '재미없어요',
        '너무 재미없다', '참 재밋네요','민수가 못 생기긴 했어요',
        '안결 혼해요'
    
]

#긍정1, 부정0
labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1])


token = Tokenizer()
token.fit_on_texts(docs) 
print(token.word_index) 

x = token.texts_to_sequences(docs)
print(x)

from keras.preprocessing.sequence import pad_sequences #Sequences로 이루어진 리스트를 넘파이 2d 배열로 바꿔주는 함수
pad_x = pad_sequences(x, padding='pre', maxlen=5 ) # 통상적으로 0은 앞에 채우도록 설정한다. 0을 뒤에 채울경우 0으로 수렴될 수 있음
                                                   # pre 앞, post 뒤
                                                   
print(pad_x)
print(pad_x.shape) #(14, 5)

word_size = len(token.word_index)
print("word_size : ", word_size) #단어사전의 갯수 : 30

print(np.unique(pad_x, return_counts = True)) #패딩처리 한 단어사전의 갯수 : 31

# array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
#        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30])


#2. 모델
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Embedding

model = Sequential() 
                                 
#표현1              #단어사전의 갯수               #(14,5) / 열
model.add(Embedding(input_dim=25, output_dim=10, input_length=5)) #input_length 명시 or 생략 둘다 가능

#표현2
# model.add(Embedding(input_dim=31, output_dim=10)) 

#표현3
# model.add(Embedding(31,1))

#표현4
# model.add(Embedding(31,3, input_length=5)) #error

#표현 불가능
# model.add(Embedding(31,10,5)) #error

model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))  

model.summary()                                  

#3.컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(pad_x, labels, epochs=15, batch_size=16)


#4.평가, 예측
acc = model.evaluate(pad_x, labels)[1]
print('acc :', round(acc,3))