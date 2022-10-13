from keras.preprocessing.text import Tokenizer

text = "나는 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다."

token = Tokenizer()
token.fit_on_texts([text]) #fit을 통해 수치화 작업 및 index 처리

print(token.word_index) #{'마구': 1, '매우': 2, '나는': 3, '진짜': 4, '맛있는': 5, '밥을': 6, '엄청': 7, '먹었다': 8}

x = token.texts_to_sequences([text])
print(x) #[[3, 4, 2, 2, 5, 6, 7, 1, 1, 1, 8]] >> 이 데이터는 수치화 작업이 되어있으나 연속성을 가진 데이터이므로 원핫인코딩 작업을 한다.

from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

x = to_categorical(x)
print(x)
print(x.shape) #(1, 11, 9) #각각의 어절이 총 11개의 열로 이루어짐

# ohe = OneHotEncoder()
# x = ohe.fit_transform(x.reshape)              
