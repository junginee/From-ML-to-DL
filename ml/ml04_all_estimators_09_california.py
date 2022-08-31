import numpy as np 
from sklearn.utils import all_estimators
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import r2_score, accuracy_score


#1.데이터
datasets = fetch_california_housing()

x = datasets['data']
y = datasets['target']


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size= 0.7,random_state=31)



#2.모델구성
# allAlgorithms = all_estimators(type_filter='classifier')  
allAlgorithms = all_estimators(type_filter='regressor')
print('allAlgorithms : ', allAlgorithms)
print('모델의 갯수 : ', len(allAlgorithms))

import warnings
warnings.filterwarnings('ignore') 

for (name, algorithm) in allAlgorithms:       
    try:                                    
        model = algorithm()
        model.fit(x_train,y_train)
    
        y_predict = model.predict(x_test)
        r2 = r2_score(y_test,y_predict)
        print(name, '의 r2 : ', r2)              
    except:                                   
        continue
   
# ARDRegression 의 r2 :  0.5948025748575727
# AdaBoostRegressor 의 r2 :  0.5738958989589461
# BaggingRegressor 의 r2 :  0.7828690455793166
# BayesianRidge 의 r2 :  0.6043334766551409
# CCA 의 r2 :  0.5672150977828783
# DecisionTreeRegressor 의 r2 :  0.5951012172164123
# DummyRegressor 의 r2 :  -7.397546157350554e-05
# ElasticNet 의 r2 :  0.42777314389319676
# ElasticNetCV 의 r2 :  0.5845873732011406
# ExtraTreeRegressor 의 r2 :  0.5615764823088671
# ExtraTreesRegressor 의 r2 :  0.8091238161000012
# GammaRegressor 의 r2 :  -7.397546157350554e-05
# GaussianProcessRegressor 의 r2 :  -2.8310806232820105
# GradientBoostingRegressor 의 r2 :  0.7816725702912718
# HistGradientBoostingRegressor 의 r2 :  0.8321018440495209
# HuberRegressor 의 r2 :  0.502504567597031
# KNeighborsRegressor 의 r2 :  0.13128111782149599
# KernelRidge 의 r2 :  0.5428908573742932
# Lars 의 r2 :  0.6043500570596736
# LarsCV 의 r2 :  0.6038441140804174
# Lasso 의 r2 :  0.28722782643043243
# LassoCV 의 r2 :  0.5886423312109831
# LassoLars 의 r2 :  -7.397546157350554e-05
# LassoLarsCV 의 r2 :  0.6038441140804174
# LassoLarsIC 의 r2 :  0.6043500570596736
# LinearRegression 의 r2 :  0.6043500570596735
# LinearSVR 의 r2 :  0.34072420272910964
# MLPRegressor 의 r2 :  0.5449177610952816
# NuSVR 의 r2 :  0.007439507638323017
# OrthogonalMatchingPursuit 의 r2 :  0.4761970991144715
# OrthogonalMatchingPursuitCV 의 r2 :  0.5965236681079236
# PLSCanonical 의 r2 :  0.3781457885308922
# PLSRegression 의 r2 :  0.5302012095600712
# PassiveAggressiveRegressor 의 r2 :  0.3946974207594781
# PoissonRegressor 의 r2 :  -7.397546157350554e-05
# RANSACRegressor 의 r2 :  0.32914310026620375
# RandomForestRegressor 의 r2 :  0.8011252009746515
# Ridge 의 r2 :  0.6043463866056734
# RidgeCV 의 r2 :  0.6043106104899904
# SGDRegressor 의 r2 :  -8.112975040805408e+29
# SVR 의 r2 :  -0.023116063295270273
# TheilSenRegressor 의 r2 :  0.47094506513500867
# TransformedTargetRegressor 의 r2 :  0.6043500570596735
# TweedieRegressor 의 r2 :  0.4959152569758437
