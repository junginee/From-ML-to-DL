from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.python.keras.optimizer_v2 import adam
import time

# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test =  train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape) # (353, 10) (89, 10)

# 2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=10))
model.add(Dense(50, activation='relu'))
model.add(Dense(30))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(20))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

# 3. compilel, fit
optimizer = adam.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, metrics=['mae'], loss='mse')

es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto', verbose=1, factor=0.5)

start = time.time()
model.fit(x_train, y_train, epochs=300, validation_split=0.2, batch_size=128, callbacks=[es,reduce_lr])
end = time.time()-start

loss, mae = model.evaluate(x_test,y_test)
y_pred = model.predict(x_test)

print('걸린 시간: ', end)
print('loss: ', loss)
print('r2: ', r2_score(y_test, y_pred))


# r2스코어 :  0.7884188322471288

# reduce lr
# 걸린 시간:  10.411244869232178
# loss:  3340.7509765625
# r2:  0.4852495435860539