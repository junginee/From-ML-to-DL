import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 데이터
train_df = pd.read_csv('house_train.csv')
test_df = pd.read_csv('house_test.csv')

train_df.head()
# set index       
train_df.set_index('Id', inplace=True)  #train 데이터 1460
test_df.set_index('Id', inplace=True)   #test 데이터 (1459,81)
len_train_df = len(train_df)
len_test_df = len(test_df)

corrmat = train_df.corr()
top_corr_features = corrmat.index[abs(corrmat["SalePrice"])>=0.3]
top_corr_features

# heatmap
plt.figure(figsize=(13,10))
g = sns.heatmap(train_df[top_corr_features].corr(),annot=True,cmap="RdYlGn")

# split y_label
train_y_label = train_df['SalePrice'] 	# target 값을 미리 분리하였음.
train_df.drop(['SalePrice'], axis=1, inplace=True)