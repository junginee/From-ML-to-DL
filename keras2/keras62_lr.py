x = 10      #임의로 변경
y = 10      #목표 결과값
w = 0.5     #임의의 가중치 초기값
lr = 0.001
epochs = 50000

for i in range(epochs):
    predict = x * w
    loss = (predict - y) ** 2
    
    print("Loss : ", round(loss, 4), '\tpredict :', round(predict, 4))
    
    up_predict = x * (w + lr)
    up_loss = (y - up_predict) ** 2
    
    down_predict = x * (w - lr)
    down_loss = (y - down_predict) ** 2
    
    if (up_loss > down_loss):
        w = w - lr
    else:
        w = w + lr    

import matplotlib.pyplot as plt        
plt.plot(x, y, 'k-')
plt.plot(2, 2, 'sk')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.show()        