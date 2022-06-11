import numpy as np
from matplotlib import pyplot as plt
import matplotlib

x_fact = np.array([8, 7.8, 7.6, 7.7, 7.7, 7, 7.3, 7.1, 6.6])
y = np.array([16.8, 16.7, 16.7, 16.6, 16.6, 16.6, 16.6, 16.6, 16.4])

F_t = np.array([[1] * len(x_fact),
                x_fact,
                x_fact ** 2,
                x_fact ** 3])
F = F_t.transpose()
A = F_t @ F
betta = np.linalg.inv(A) @ F_t @ y.transpose()


def model_1(x):
    return sum([betta[i] * x ** i for i in range(len(betta))])

def found_confidence_interval(t, point):
    """Print confidence interval for mean value and value of output value
    param t: value of t from P{|xi|<t}=gamma and xi~St_(n-m)  """
    x_vec_t = np.array([point ** i for i in range(4)])
    x_vec = x_vec_t.transpose()
    model = sum([betta[i] * x_vec[i] for i in range(len(x_vec))])
    a = model - t * ((x_vec_t @ np.linalg.inv(A) @ x_vec) * denominator) ** 0.5
    b = model + t * ((x_vec_t @ np.linalg.inv(A) @ x_vec) * denominator) ** 0.5
    print('(', np.round(a, 4), ';', np.round(b, 4), ')')
    a1 = model - t * ((x_vec_t @ np.linalg.inv(A) @ x_vec+1) * denominator+1) ** 0.5
    b1 = model + t * ((x_vec_t @ np.linalg.inv(A) @ x_vec+1) * denominator) ** 0.5
    print('(', np.round(a1, 4), ';', np.round(b1, 4), ')')


numerator = 1 / (len(x_fact) - 1) * sum([(i - y.mean()) ** 2 for i in y])
denominator = (1 / (len(x_fact) - len(betta))) * np.linalg.norm((y - F @ betta)) ** 2
print(denominator)
print("statistic's value for check our model: ", numerator / denominator)
valuable_param = betta[3] / ((denominator * np.linalg.inv(A)[3][3]) ** 0.5)
print("gamma-value: ", valuable_param)
print("betta-vector:")
print(*betta, sep='\n')
fig = plt.figure()
# fig.suptitle("Sample's visualisation")
ax = fig.add_subplot()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid()
ax.scatter(x_fact, y)
ax.plot(np.arange(6.6, 8, 0.01), model_1(np.arange(6.6, 8, 0.01)), color='orange')
# ax.plot(np.arange(6.6, 8, 0.01), 0.2051 * np.arange(6.6, 8, 0.01) + 15.0995, color='green')
plt.show()


t = 2.571


found_confidence_interval(t, 7.5)


