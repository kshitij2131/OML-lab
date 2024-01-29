import numpy as np 
from numpy import linalg
import pandas as pd

r = 8.0
epsilon = 0.01
maxIter = 500
beta1 = 1e-4
beta2 = 0.9

A, y, x = open()
numParams = 2

def open():
    path = "2_col_revised.xlsx"
    dt = pd.read_excel(path)
    B=dt.values
    x,y=B[:,0],B[:,1]
    
    x[-1] = r
    y[-1] = 2*r + 3.5

    x = x.astype(float)
    y = y.astype(float)
    
    y = np.array(y).reshape(-1, 1)
    numPnts = len(x)
    A=np.column_stack((x,np.ones((numPnts,1),dtype=float)))
    return (A, y, x)


def fun(params):
    numPnts = params.shape[0]
    P = np.dot(A, params) - y
    val = 0.0
    for i in range(numPnts):
        val += (P[i][0]**2)
    
    val /= (2*numPnts)
    return val

def grad(params):
    numParams = params.shape[0]
    F = fun(params)
    h, g = 1e-5, np.zeros(numParams)

    for i in range(numParams):
        params[i][0] += h
        f = fun(params)
        g[i] = (f - F)/h
        params[i][0] -= h
    return g.reshape(-1, 1)


def armijo(x, alpha, d, gradf):
    lhs = fun(x+alpha*d)
    rhs = fun(x) + alpha*beta1*gradf
    return lhs <= rhs

def wolfe(x, alpha, d, gradf):
    lhs = np.dot(grad(x+alpha*d).T, d)
    rhs = beta2*np.dot(gradf.T, d)
    return lhs >= rhs

if __name__ == '__main__':
    
    params = np.ones((numParams, 1))
    iter = 0
    gradf = grad(params)
    xk = params

    while iter < maxIter and linalg.norm(gradf) >= epsilon:
        dk = -gradf
        alpha = 1
        while (not armijo(xk, alpha, dk, gradf)) and (not wolfe(xk, alpha, dk, gradf)):
            alpha = alpha/2
        
        xk = xk + alpha*dk
        gradf = grad(xk)
        iter += 1
    
    print("number of iterations = ", iter)
    print("optimal solution at x* = ", xk)
    print("minimum value of loss function = ", fun(xk))
    print("norm of gradient at x* = ", grad(xk))



    
    