      
#1.데이터
import numpy as np  
x = np.array([1,2,3])       
y = np.array([1,2,3])                                                                                                                                                                          
                                                                                                                        
#2.모델구성                                                                                             
from tensorflow.keras.models import Sequential         
from tensorflow.keras.layers import Dense                            
                                                                      
model = Sequential()                                      
model.add(Dense(4, input_dim=1))   
model.add(Dense(5))
model.add(Dense(3)) 
model.add(Dense(3))                                             
model.add(Dense(1)) 
            
#3. 컴파일, 훈련                       
model.compile(loss='mse',optimizer='adam') 
model.fit(x,y,epochs=600)

#4. 평가, 예측
loss = model.evaluate(x, y) 
print('loss :', loss) 
 
result = model.predict([4])
print('4의 예측값 : ', result)              
                                    
