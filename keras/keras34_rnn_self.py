 
import numpy as np

dataset=np.array(range(1,11)) #(10,)
#array에서 앞에서부터 4개(처음엔 1~4) 값을 가져와서 그 다음값(5)를 y로 설정하게
#split을 한다.

def split(dataset,time_steps):
    xs=list()
    ys=list()
    for i in range(0,len(dataset)-time_steps):
        x=dataset[i:i+time_steps]
        y=dataset[i+time_steps]
        # print(f"{x}:{y}")
        xs.append(x)
        ys.append(y)
    return np.array(xs),np.array(ys)

xs,ys=split(dataset,4)

print(f"xs:{xs}\n ys:{ys}")