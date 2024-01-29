import numpy as np
from cvxopt import matrix, solvers

c, b = np.array([[3], [-4]]), np.array(
    [[12], [20], [-5], [0]])
A = np.array([[1, 3], [2, -1], [-1, 4], [-1, 0]])

soln = solvers.lp(matrix(c, tc='d'), matrix(A, tc='d'), matrix(b, tc='d'))

print(np.round_(soln['x'], decimals=5), soln['primal objective'])