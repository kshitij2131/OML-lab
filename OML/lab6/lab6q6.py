import numpy as np
from numpy import linalg
import pandas as pd
import mpmath as mpm


max_iter = 5000
epsilon = 1e-4
r = 8.0
lmbda = abs(r/10 - 5)


pi = np.pi

n = 8
file = 'diabetes2.csv'
df = pd.read_csv(file)
A = df.iloc[:, :n].to_numpy()
y = df.iloc[:, n].to_numpy().reshape(-1, 1)
A = A.T
m = A.shape[1]
# print(A.shape)

minmax_val = np.zeros((n, 2))
for i in range(n):
    minmax_val[i][0] = np.min(A[i, :])
    minmax_val[i][1] = np.max(A[i, :])
    
    if minmax_val[i][1] > minmax_val[i][0]:
        normalized_row = (A[i, :] - minmax_val[i][0]) / (minmax_val[i][1] - minmax_val[i][0])
        A[i, :] = normalized_row


def p(params):
    dp = np.dot(A.T, params)
    return 1/(1+np.exp(dp))


def lossFun(params):
    pf = p(params)
    val = 0.0
    for i in range(m):
        if y[i][0] == 0:
            val += np.log(1 - pf[i][0])
        if y[i][0] == 1:
            val += np.log(pf[i][0])
        
    val *= (-1/m)
    val *= (1/lmbda)
    
    return val

def grad(params):
    numParams = params.shape[0]
    
    F = lossFun(params)

    h, g = 1e-5, np.zeros(numParams)

    for i in range(numParams):
        params[i][0] += h
        f = lossFun(params)
        g[i] = (f - F)/h
        params[i][0] -= h
    
    return g.reshape(-1, 1)



def eval(yt):
    score = 0
    for i in range(m):
        if round(yt[i][0]) == y[i][0]:
            score += 1
        # print(yt[i][0], y[i][0])
    print("accuracy :", score/m)
    print("\n")


def prox_op(alphak, x_cap):
    numParams = x_cap.shape[0]
    prox_op_res = np.zeros((numParams, 1))
    for i in range(numParams):
        if x_cap[i] > alphak:
            prox_op_res[i][0] = (x_cap[i][0] - alphak)
        if x_cap[i][0] < -alphak:
            prox_op_res[i][0] = (x_cap[i][0] + alphak)
    return prox_op_res


def comp_opt_prob(x):

    xk = x
    k = 0
    iter = 0

    while iter < max_iter:
        alphak = 1/(r + k)
        x_cap = xk - alphak*grad(xk)
        # print(x_cap)
        prox = prox_op(alphak, x_cap)
        nrm = np.linalg.norm(xk - prox)
        if nrm < epsilon:
            break
        xk = prox
        k += 1
        iter += 1

    return (xk, iter)


    




if __name__ == '__main__':
    
    params = np.ones((n, 1))
    x_star, iter = comp_opt_prob(params)

    print("----SOLUTION USING PROXIMAL GRADIENT METHOD AND L1 REGULARIZATION----")
    print("number of iterations = ", iter)
    print("minimum is obtained at x* =  ", x_star)
    print("minimum value of objective function at x* = ", lmbda*lossFun(x_star) + lmbda*np.linalg.norm(x_star, ord=1))
    eval(p(x_star))



    



    







    


