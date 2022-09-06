import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor


# 1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=1234, train_size=0.8, stratify=y)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# 2. 모델
model = XGBClassifier(n_estimators=1000,
              learning_rate=1,
              max_depth=2,
              gamma=0,
              min_child_weight=1,
              subsample=1,
              colsample_bytree=0.5,
              colsample_bylevel=1,
              colsample_bynode=1,
              reg_alpha=0.01,
              reg_lambd=1,
              tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0, random_state=1234
              )


model.fit(x_train, y_train, early_stopping_rounds=10, 
          eval_set=[(x_train, y_train), (x_test, y_test)],
          eval_metric=['logloss'])


print('테스트 스코어: ', model.score(x_test, y_test))

print('-----------------------------------------------------------------------')
hist = model.evals_result()
print(hist)


import matplotlib.pyplot as plt
print(hist.keys()) 
# dict_keys(['validation_0', 'validation_1'])

# plt.figure(figsize=(10,10))
# for i in range(len(hist.keys())):
#     plt.subplot(len(hist.keys()),1, i+1) 
#     # nrows = len(hist.keys()), ncols=1, index = i+1  
#     plt.plot(hist['validation_'+str(i)]['logloss'])
#     plt.xlabel('n_estimators')
#     plt.ylabel('evals_result')
#     plt.title('validation_'+str(i))
# plt.show()

# plt.figure(figsize=(10,10))
# plt.subplot(2,1,1)
# plt.plot(hist['validation_0']['logloss'])

# plt.subplot(2,1,2)
# plt.plot(hist['validation_1']['logloss'])
import matplotlib.pyplot as plt

def plot_graphs(history, string):
  plt.plot(history.hist[string])
  plt.plot(history.hist['val_'+string])
  plt.xlabel("n_estimators")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
# plot_graphs(hist, "acc")
# plot_graphs(hist, "loss")
