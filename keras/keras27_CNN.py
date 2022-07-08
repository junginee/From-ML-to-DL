#https://keras.io/api/layers/convolution_layers/convolution2d/

#노트필기 참고

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten #이미지는 2D

model = Sequential()
#model.add(Dense(units=10, input_shape=(3,)))  #input_shape=(batch_size, input_dim )
#model.summary() #(input_dim + bias) * units = summary Param 갯수 (Dense 모델)

model.add(Conv2D(filters=10, kernel_size = (3,3),  #출력 : ( N, 6, 6, 10 )
                                  input_shape = (8,8,1) )) #input_shape=(batch_size, row, columns, channels) 
model.add(Conv2D(7,(2,2), activation = 'relu'))   #출력 : ( N, 5, 5, 7 )
model.add(Flatten()) #출력 : (N, 28)

# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(10, activation='softmax'))


#input_shape = N, 5, 5, 1 (N장의 5*5 흑백 이미지) / 1은 흑백, 3은 컬러
#커널 사이즈는 이미지 자르는 규격
#filters 갯수만큼, 즉 아웃풋 노드 수 만큼 증폭된다.

model.summary() # ( (kernel_size *channels ) + bias) ) * filters = summary Param 갯수 (CNN 모델)

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 4, 4, 10)          50             #CNN 파라미터 구하는 공식 : (커널사이즈(2*2) * 채널수(1) + bias node(1)) * output 갯수(10)) = 50
=================================================================
Total params: 50
Trainable params: 50
Non-trainable params: 0
_________________________________________________________________


'''


