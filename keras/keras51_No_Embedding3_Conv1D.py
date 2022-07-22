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


token = Tokenizer() #토큰화 함수 지정
token.fit_on_texts(docs) #토큰화 함수에 문장적용
print(token.word_index) #각 단어에 매겨진 인덱스 값 출력

x = token.texts_to_sequences(docs) #텍스트 안의 단어들을 숫자의 시퀀스의 형태로 변환
print(x)

from keras.preprocessing.sequence import pad_sequences #Sequences로 이루어진 리스트를 넘파이 2d 배열로 바꿔주는 함수
pad_x = pad_sequences(x, padding='pre', maxlen=5, truncating='pre' )
# truncating 
# 통상적으로 0은 앞에 채우도록 설정한다. 0을 뒤에 채울경우 0으로 수렴될 수 있음
# pre 앞, post 뒤
                                                   
print(pad_x)
print(pad_x.shape) #(14, 5)
pad_x = pad_x.reshape(14,5,1)

word_size = len(token.word_index)
print("word_size : ", word_size) #단어사전의 갯수 : 30

print(np.unique(pad_x, return_counts = True)) #패딩처리 한 단어사전의 갯수 : 31


#2. 모델
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Embedding, Conv1D

model = Sequential() 
                                 
#표현1                      
# model.add(Embedding(input_dim=31, output_dim=10, input_length=5)) #input_length 명시 or 생략 둘다 가능
model.add(Conv1D(32,2, input_shape = (5,1)))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='sigmoid'))  
model.add(Dense(10, activation='sigmoid'))  
model.add(Dense(5, activation='sigmoid'))  
model.add(Dense(1, activation='sigmoid'))  

model.summary()                                  

#3.컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(pad_x, labels, epochs=100, batch_size=5)


#4.평가, 예측
acc = model.evaluate(pad_x, labels)[1]
print('acc :', round(acc,3))


#긍정1, 부정0
#########[실습]##########
x_predict = ['나는 형권이가 정말 재미없다 너무 정말']

token.fit_on_texts(x_predict) #토큰화 함수에 문장적용
print(token.word_index)
# {'너무': 1, '참': 2, '재미없다': 3, '정말': 4, '재밌어요': 5, 
#  '최고에요': 6, '잘': 7, '만든': 8, '영화예요': 9, '추천하고': 10, 
#  '싶은': 11, '영화입니다': 12, '한': 13, '번': 14, '더': 15, 
#  '보고': 16, '싶네요': 17, '글세요': 18, '별로에요': 19, '생각보다': 20, 
#  '지루해요': 21, '연기가': 22, '어색해요': 23, '재미없어요': 24, '재밋네요': 25, 
#  '민수가': 26, '못': 27, '생기긴': 28, '했어요': 29, '안결': 30, 
#  '혼해요': 31, '나는': 32, '형권이가': 33}


x_predict = token.texts_to_sequences(x_predict)
x_predict = x_predict.reshape(6,1,1)
print(x_predict) #[[32, 33, 4, 3, 1, 4]]

result = model.predict(x_predict)
print(result)

if result >= 0.5 :
    print("긍정")
else :
    print("부정")
    
    
