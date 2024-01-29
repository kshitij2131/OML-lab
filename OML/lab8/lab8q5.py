#kshitij jaiswal
#b20cs028

import numpy as np 
from numpy import linalg
import pandas as pd
import random
import matplotlib.pyplot as plt

r = 8.0
maxIter = 1e5
numParams = 3


def open():
    path = "2_col_revised.xlsx"
    dt = pd.read_excel(path)
    B=dt.values
    x,y=B[:,0],B[:,1]
    
    x[-1] = r
    y[-1] = 2*r + 3.5
    x = x.astype(float)
    y = y.astype(float)
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    numPnts = len(x)
    A=np.column_stack((x*x,x,np.ones((numPnts,1),dtype=float)))
    return (A, x, y)

A, x, y = open()
numPnts = A.shape[0]
# print(A, y)


def fun(params):
    P = np.dot(A, params) - y
    val = 0.0
    for i in range(numPnts):
        val += (P[i][0]**2)
    
    val /= (2*numPnts)
    return val

def grad(params, i):
    Ai = np.array(A[i]).reshape(-1,1)
    gradVal = np.dot(np.dot(Ai, Ai.T), params) - Ai*y[i][0]
    gradVal /= numPnts
    return gradVal


if __name__ == '__main__':
    
    params = np.ones((numParams, 1))
    iter = 0

    xbest = params
    fbest = fun(xbest)

    
    alpha = 0.001

    while iter < maxIter:
        i = random.randint(0, numPnts-1)
        gradf = grad(xbest, i)
        dk = -gradf

        xcand = xbest + alpha*dk
        if fun(xcand) < fbest:
            xbest = xcand
            fbest = fun(xcand)
            iter += 1

    
    
        
    
    print("number of iterations = ", iter)
    print("optimal solution at x* = ", xbest)
    print("minimum value of loss function = ", fbest)


    T = np.dot(A, xbest)
    plt.plot(x, y, 'b.')
    plt.plot(x, T, 'r.')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


    
    