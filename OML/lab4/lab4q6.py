import numpy as np
from cvxopt import matrix, solvers
n = 4

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

c = np.array([[20], [28], [19], [13], [15], [30], [31], [28], [40], [21], [20], [17], [21], [28], [26], [12]])




# print(A)
# print(B)


soln = solvers.lp(matrix(c, tc='d'), matrix(A, tc='d'), matrix(b, tc='d'))

print(np.round_(soln['x'], decimals=5), soln['primal objective'])  
