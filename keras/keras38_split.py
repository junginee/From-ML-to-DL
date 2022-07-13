import numpy as np

a = np.array(range(1,11))
size = 3

def split_x(dataset,size):
    aaa = []
    for i in range(len(dataset)-size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, size)
print(bbb)
print(bbb.shape)

x = bbb[:, :-1] #x값 뒤에서 하나씩 추출한다.
y = bbb[:, 2] #추출한 값을 y에 넣어준다.

print(x,y)
print(x.shape, y.shape)