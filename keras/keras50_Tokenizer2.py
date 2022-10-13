from keras.preprocessing.text import Tokenizer

text1 = "나는 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다."
text2 = "나는 지구용사 이재근이다. 멋있다. 또 또 얘기해봐"


token = Tokenizer()
token.fit_on_texts([text1, text2]) #fit을 통해 수치화 작업 및 index 처리

print(token.word_index) 

x = token.texts_to_sequences([text1,text2])
print(x) #>> 이 데이터는 수치화 작업이 되어있으나 연속성을 가진 데이터이므로 원핫인코딩 작업

from tensorflow.python.keras.utils.np_utils import to_categorical

x_new = x[0] + x[1] #차원 변화
print(x_new)

x_new = to_categorical(x_new)
print(x_new)
print(x_new.shape) #(18, 14)

# ohe = OneHotEncoder()
# x = ohe.fit_transform(x.reshape)


###################################################################################

import numpy as np

text1 = "나는 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다."
text2 = "나는 지구용사 이재근이다. 멋있다. 또 또 얘기해봐"


token = Tokenizer()
token.fit_on_texts([text1, text2]) 

x = token.texts_to_sequences([text1,text2])
print(x)

def one_hot_encoding(word, word_dict):
    one_hot_vector = np.zeros(len(word_dict)) # 단어사전 길이만크의 0벡터 생성
    one_hot_vector[word_dict[word]] = 1 # 해당하는 단어의 index 자리에 1 부여
    return one_hot_vector

word_list = []
for w in token:
    word_list.append(one_hot_encoding(w, word_dict))
print(np.array(word_list))
