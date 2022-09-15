import numpy as  np
import matplotlib.pyplot as plt


def elu(x, a=1):
    return (np.where(x>=0, x, 0.1*(np.exp(x)-1)))

elu2 = lambda x : np.where(x>=0, x, 0.1*(np.exp(x)-1))

x = np.arange(-5, 5, 0.1)
y = elu2(x)

plt.plot(x,y)
plt.grid()
plt.show()