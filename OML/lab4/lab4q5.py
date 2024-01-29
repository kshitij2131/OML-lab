import numpy as np
from cvxopt import matrix, solvers
n = 3
m = 5

A = np.zeros(((n<<1)+m+(n*m), n*m))


for i in range(0, n<<1, 2):
    for j in range((i>>1)*m, ((i>>1)+1)*m):
        # print(i, j)
        A[i][j] = 1
        A[i+1][j] = -1
        

for i in range(n<<1, (n<<1)+m):
    for j in range(0, n*m, m):
        A[i,j + (i-(n<<1))%m] = 1
        

for i in range((n<<1)+m, (n<<1)+m+(n*m)):
    A[i, i-((n<<1)+m)]=-1

# for i in range((n<<1)+m+(n*m)):
#     for j in range(n*m):
#         print(int(A[i, j]), end=" ")
#     print("\n")

# print(A)

b = np.array([[8], [-8], [12], [-12],[14], [-14], [7], [5], [6], [8], [8]])
b = np.row_stack((b, np.zeros((n*m, 1))))
# print(b)


c = np.array([[4], [2], [3], [2], [6], [5], [4], [5], [2], [1], [6], [5], [4], [7], [7]])




# # print(A)
# # print(B)


soln = solvers.lp(matrix(c, tc='d'), matrix(A, tc='d'), matrix(b, tc='d'))

print(np.round_(soln['x'], decimals=5), soln['primal objective'])  
