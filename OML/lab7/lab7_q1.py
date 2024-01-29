import numpy as np
import matplotlib.pyplot as plt

#Roll No. is B20CS073
r = 3

# objective function evaluated at x
def f(x):
    return max((x[0] - 2)**2 + (x[1] + 2)**2, (x[0])**2 + 8*x[1])

# subgradient of the objective function evaluated at x
def subgradient_f(x):
    if f(x) == (x[0] - 2)**2 + (x[1] + 2)**2:
        subgrad = 2 * (x - np.array([2, -2]))
    else:
        subgrad = np.array([2*x[0], 8])
    return subgrad


x = np.array([0.0, 0.0])
alpha = 1/5

iterations = []
fbest_values = []

max_iterations = 500

# Finding optimal value using Subgradient Descent Method
for k in range(1, max_iterations + 1):
    subgrad = subgradient_f(x)
    x = x - alpha*subgrad
    fbest = f(x)

    if(k>1 and fbest_values[len(fbest_values)-1]<= fbest ):
      fbest = fbest_values[len(fbest_values)-1]

    iterations.append(k)
    fbest_values.append(fbest)

plt.plot(iterations, fbest_values)
plt.xlabel('Iteration')
plt.ylabel('f_best(x)')
plt.title('Subgradient Descent Method')
plt.show()