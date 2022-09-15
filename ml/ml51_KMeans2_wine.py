import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

datasets = load_wine()
print(np.unique(datasets.target)) #[0 1 2]

df = pd.DataFrame(datasets.data, columns=[datasets.feature_names])
print(df)

kmeans = KMeans(n_clusters=3, random_state=1234)
#n_clusters = y 라벨의 갯수(0,1,2)

kmeans.fit(df)
print(kmeans.labels_)
print(datasets.target) 

df['cluster'] = kmeans.labels_
df['target'] = datasets.target

# accuracy_score 구하기
score =  accuracy_score(kmeans.labels_,datasets.target )
print('score :', round(score,5))
