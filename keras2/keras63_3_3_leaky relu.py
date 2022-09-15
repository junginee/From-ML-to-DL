import numpy as  np
import matplotlib.pyplot as plt


def leaky_relu(x):
    return np.maximum(0.1*x, x)

leaky_relu2 = lambda x : np.maximum(0.1*x, x)

x = np.arange(-5, 5, 0.1)
y = leaky_relu2(x)

plt.plot(x,y)
plt.grid()
plt.show()