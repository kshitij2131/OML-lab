import numpy as np
from numpy import linalg
import pandas as pd
import mpmath as mpm

delta = 0.01
maxIter = 5000
r0 = 0.50  
beta1 = 1e-4
beta2 = 0.9
epsilon = 1e-4
lmbda = 0.001

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

    
    # add L2 regularization..
    val += (0.5*lmbda*linalg.norm(params))
    
    return val

def gradFx(params):
    numParams = params.shape[0]
    
    F = lossFun(params)

    h, g = 1e-5, np.zeros(numParams)

    for i in range(numParams):
        params[i][0] += h
        f = lossFun(params)
        g[i] = (f - F)/h
        params[i][0] -= h
    
    return g.reshape(-1, 1)

def hessianFx(params):
    h = 1e-5
    numParams = params.shape[0]
    H = np.matrix(np.zeros((numParams, numParams)))
    F = lossFun(params)
   
    for i in range(numParams):
        params[i][0] += h
        fxi = lossFun(params)

        for j in range(i+1):
            params[j][0] += h

            params[i][0] -= h
            fxj = lossFun(params)
            params[i][0] += h

            H[i,j] = (lossFun(params) - fxi - fxj + F)/(h**2)
            H[j,i] = H[i,j]
            params[j][0] -= h
        params[i][0] -= h
    
    '''
    #adding the regularization parameter..
    lmbda = min(np.linalg.eig(H)[0])
    if lmbda > delta:
        lmbda = 0
    else:
        lmbda = delta-lmbda
    for i in range(numParams):
        H[i,i] += lmbda
    
    '''

    return H


def armijo(xk, alphak, dk, gradFxk):
    lhs = lossFun(xk + alphak*dk)
    rhs = lossFun(xk) + alphak*beta1*np.dot(gradFxk.T, dk)
    if lhs <= rhs:
        return True
    return False

def wolfe(xk, alphak, dk, gradFxk):
    lhs = np.dot(gradFx(xk + alphak*dk).T, dk)
    rhs = beta2*np.dot(gradFxk.T, dk)
    if lhs >= rhs:
        return True
    return False

def eval(yt):
    score = 0
    for i in range(m):
        if round(yt[i][0]) == y[i][0]:
            score += 1
        # print(yt[i][0], y[i][0])
    print("accuracy :", score/m)
    print("\n")


def steepest_descent(params, step):
    print("...RUNNING STEEPEST DESCENT...")
    
    iter = 0
    xk = params
    gradFxk = gradFx(xk)
    
    while linalg.norm(gradFxk) >= epsilon and iter < maxIter:
        if iter%step == 0:
            print("iteration ", iter)
            print("gradient norm ", linalg.norm(gradFxk))

        dk = -1 * gradFxk
    
        alpha = 1.00
        while armijo(xk, alpha, dk, gradFxk) == False or wolfe(xk, alpha, dk, gradFxk):
            alpha = alpha*r0


        xk = xk + alpha*dk
        gradFxk = gradFx(xk)
        iter += 1

    print("STEEPEST DESCENT METHOD")
    print("number of iterations :", iter)
    print("optimal solution for x :", xk)
    print("minimum value of loss function :", lossFun(xk))

    eval(p(xk))


def mirror_descent(params, step):
    print("...RUNNING MIRROR DESCENT...")

    Q = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1):
            if i == j:
                Q[i, i] = 0.1
            else:
                Q[i, j] = 0
                Q[j, i] = Q[i, j]


    iter = 0
    xk = params
    gradFxk = gradFx(xk)
    
    while linalg.norm(gradFxk) >= epsilon and iter < maxIter:
        if iter%step == 0 :
            print("iteration ", iter)
            print("gradient norm ", linalg.norm(gradFxk))

        dk = -1 * linalg.solve(Q, gradFxk)
    
        alpha = 1.00
        while armijo(xk, alpha, dk, gradFxk) == False or wolfe(xk, alpha, dk, gradFxk):
            alpha = alpha*r0

        xk = xk + alpha*dk
        gradFxk = gradFx(xk)
        iter += 1
    
    print("MIRROR DESCENT METHOD")
    print("number of iterations :", iter)
    print("optimal solution for x :", xk)
    print("minimum value of loss function :", lossFun(xk))

    eval(p(xk))

    

def newton_method(params):
    print("...RUNNING NEWTON'S METHOD...")
    
    iter = 0
    xk = params
    gradFxk = gradFx(xk)
    hessFxk = hessianFx(xk)
    # A = np.vstack((-np.identity(numParams), np.identity(numParams)))
    
    while linalg.norm(gradFxk) >= epsilon and iter < maxIter:
        # b = np.append(xk-(-1.0), 1.0-xk)
        # b[0] += (-1.0)
        # b[0] -= 0.001
    
        # sol = solvers.qp(matrix(hessFxk, tc = 'd'), matrix(gradFxk), matrix(A), matrix(b))
        # dk = np.array(sol['x'])
        dk = -linalg.solve(hessFxk, gradFxk)
        xk = xk + dk
                    
        gradFxk = gradFx(xk)
        hessFxk = hessianFx(xk)
        iter += 1
    
    print("NEWTON'S METHOD")
    print("number of iterations :", iter)
    print("optimal solution for x :", xk)
    print("minimum value of loss function :", lossFun(xk))

    eval(p(xk))

    




if __name__ == '__main__':
    
    params = np.zeros((n, 1))


    steepest_descent(params, 100)
    mirror_descent(params, 50)
    newton_method(params)



    







    


