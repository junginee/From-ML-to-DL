import numpy as np
import pandas as pd

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
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

kmeans.fit(df) # x를 훈련시켜 y(라벨)을 찾는 방식

print(kmeans.labels_) #훈련을 통해 찾은 y라벨
print(datasets.target) #데이터셋에 들어있는 target

df['cluster'] = kmeans.labels_
df['target'] = datasets.target

# accuracy_score 구하기
score =  accuracy_score(kmeans.labels_,datasets.target )
print('score :', round(score,5))
