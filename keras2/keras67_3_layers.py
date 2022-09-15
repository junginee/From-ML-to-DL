from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))



#1.
# model.trainable=False
# Total params: 17
# Trainable params: 0
# Non-trainable params: 17
# __________________________
#2.
# for layer in model.layers:
#     layer.trainable = False

# model.layers[0].trainable = False #dense 1번째 레이어
# Total params: 17
# Trainable params: 11
# Non-trainable params: 6

# model.layers[1].trainable = False #dense 2번째 레이어
# Total params: 17
# Trainable params: 9
# Non-trainable params: 8

# model.layers[2].trainable = False #dense 3번째 레이어
# Total params: 17
# Trainable params: 14
# Non-trainable params: 3


model.summary()
print(model.layers)