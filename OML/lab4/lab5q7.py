import numpy as np
from cvxopt import matrix, solvers
n = 5

A = np.zeros(((n<<2) + n*n, n*n))
# print(A)

print(n<<1, n<<2, n<<3)

for i in range(0, n<<1, 2):
    for j in range((i>>1)*n, ((i>>1)+1)*n):
        # print(i, j)
        A[i][j] = 1
        A[i+1][j] = -1
        

for i in range(n<<1, n<<2, 2):
    for j in range(0, n*n, n):
        # print(i, j + (i>>1)%n)
        A[i,j + (i>>1)%n] = 1
        A[i+1,j + (i>>1)%n] = -1
        

for i in range(n<<2, (n<<2) + n*n, 1):
    A[i, i-(n<<2)]=-1

# for i in range((n<<2)+(n*n)):
#     for j in range(n*n):
#         print(int(A[i, j]), end=" ")
#     print("\n")

b = np.zeros(((n<<2) + n*n, 1))
for i in range(n<<2):
    if i&1:
        b[i, 0] = -1
    else:
        b[i, 0] = 1

c = np.array([[37.7], [32.9], [33.8], [37.0], [35.4], [43.4], [33.1], [42.2], [34.7], [41.8], [33.3], [28.5], [38.9], [30.4], [33.6], [29.2], [26.4], [29.6], [28.5], [31.1]])
c = np.row_stack((c, np.zeros((n, 1))))



# print(A)
# print(b)

# print(c)

soln = solvers.lp(matrix(c, tc='d'), matrix(A, tc='d'), matrix(b, tc='d'))

print(np.round_(soln['x'], decimals=5), soln['primal objective'])  
