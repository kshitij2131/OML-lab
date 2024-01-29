import numpy as np
from cvxopt import matrix, solvers


A = np.array([[1, 2], [-4, -3], [3, 1], [-3, -1], [1, 0], [-1, 0], [0, -1]])
b1 = np.array([[3], [-6], [3], [-3], [1], [-15/17], [0]])
c1 = np.array([5, 2])

b2 = np.array([[3], [-6], [3], [-3], [15/17], [0], [0]])
c2 = np.array([3, 7])

soln1 = solvers.lp(matrix(c1, tc='d'), matrix(A, tc='d'), matrix(b1, tc='d'))
soln2 = solvers.lp(matrix(c2, tc='d'), matrix(A, tc='d'), matrix(b2, tc='d'))

# print(np.round_(soln1['x'], decimals=5), soln1['primal objective']) --> gives unbounded solution...
print(np.round_(soln2['x'], decimals=5), soln2['primal objective'])  
