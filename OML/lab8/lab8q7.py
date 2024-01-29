#kshitij jaiswal
#b20cs028

import numpy as np
from numpy import linalg
import pandas as pd
import mpmath as mpm
import random
import matplotlib.pyplot as plt


r = 8.0
k = 10


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
        
    val *= (-1/(2*m))
    return val

def eval(yt):
    score = 0
    for i in range(m):
        if round(yt[i][0]) == y[i][0]:
            score += 1
        # print(yt[i][0], y[i][0])
    return score/m


def grad_stoch(params, i):
    Ai = np.array(A[:,i]).reshape(-1,1)
    gradVal = np.zeros((n, 1))
    if y[i][0] == 0:
        gradVal += (Ai/(1+np.exp(np.dot(Ai.T, params))))
    else:
        gradVal += ((-np.exp(np.dot(Ai.T, params))*Ai)/(1+np.exp(np.dot(Ai.T, params))))
    gradVal /= (-m)
    return gradVal

def grad_minibatch(params, rndPnts):
    gradVal = np.zeros((n, 1))
    for i in rndPnts:
        Ai = np.array(A[:,i]).reshape(-1,1)
        if y[i][0] == 0:
            gradVal += (Ai/(1+np.exp(np.dot(Ai.T, params))))
        else:
            gradVal += ((-np.exp(np.dot(Ai.T, params))*Ai)/(1+np.exp(np.dot(Ai.T, params))))
    gradVal /= (-m*k)
    return gradVal


def minibatch_grad(params, maxIter):
    iter=0
    xbest = params
    fbest = lossFun(xbest)
    alpha = 15

    while iter < maxIter:
        rndPnts = random.sample(range(m), k)
        gradf = grad_minibatch(xbest, rndPnts)
        dk = -gradf

        xcand = xbest + alpha*dk
        if lossFun(xcand) < fbest:
            xbest = xcand
            fbest = lossFun(xcand)
            iter += 1
        # print(iter)

    print()
    print("-------SOLUTION USING MINI-BATCH GRADIENT-------")
    print("number of iterations = ", iter)
    print("optimal solution at x* = ", xbest)
    print("minimum value of loss function = ", fbest)
    print("accuracy = ", eval(np.dot(A.T, xbest)))
    print()

def stochastic_grad(params, maxIter):
    iter = 0
    xbest = params
    fbest = lossFun(xbest)
    alpha = 4

    while iter < maxIter:
        i = random.randint(0, m-1)
        gradf = grad_stoch(xbest, i)
        dk = -gradf

        xcand = xbest + alpha*dk
        if lossFun(xcand) < fbest:
            xbest = xcand
            fbest = lossFun(xcand)
            iter += 1
        # print(iter)
    print()
    print("-------SOLUTION USING STOCHASTIC GRADIENT-------")
    print("number of iterations = ", iter)
    print("optimal solution at x* = ", xbest)
    print("minimum value of loss function = ", fbest)
    print("accuracy = ", eval(np.dot(A.T, xbest)))
    print()




if __name__ == '__main__':
    
    params = np.ones((n, 1))
    stochastic_grad(params, 500)
    minibatch_grad(params, 500)
    





    



    







    


