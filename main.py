import numpy as np
from matplotlib import pyplot as plt

x_fact = np.array([8, 7.8, 7.6, 7.7, 7.7, 7, 7.3, 7.1, 6.6])
y = np.array([16.8, 16.7, 16.7, 16.6, 16.6, 16.6, 16.6, 16.6, 16.4])

fig = plt.figure()
fig.suptitle("Sample's visualisation")
ax = fig.add_subplot()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.scatter(x_fact, y)
ax.grid()
plt.show()
