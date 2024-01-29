import numpy as np
from numpy import linalg


r = 8.00
delta = 0.01
maxIter = 500
epsilon = 1e-4


def fun1(params):
    x1 = params[0][0]
    x2 = params[1][0]

    return (x1**2) + (x2**2) - 2

def fun2(params):
    x1 = params[0][0]
    x2 = params[1][0]

    return (np.e)**(x1-1) + (x2**3) - 2

def gradFx(params, fx):
    numParams = params.shape[0]
    
    F = fx(params)

    h, g = 1e-5, np.zeros(numParams)

    for i in range(numParams):
        params[i][0] += h
        f = fx(params)
        g[i] = (f - F)/h
        params[i][0] -= h
    
    return g.reshape(-1, 1)

def jacobianFx(params):
    gradFx1 = gradFx(params, fun1)
    gradFx2 = gradFx(params, fun2)

    j = np.vstack((gradFx1.T, gradFx2.T))
    return j



if __name__ == '__main__':

    params = np.array([[2+(r/10)], [2+(r/10)]])
    numParams = params.shape[0]


    #using newton's method..
    
    iter = 0
    xk = params
    Fxk = np.vstack((fun1(xk), fun2(xk)))
    jacFxk = jacobianFx(xk)

    
    while linalg.norm(Fxk) >= epsilon and iter < maxIter:
        dk = -np.linalg.solve(jacFxk, Fxk)
        xk = xk + dk
        Fxk = np.vstack((fun1(xk), fun2(xk)))
        jacFxk = jacobianFx(xk)
        iter += 1

    print("optimal solution for x (using newton's method): ", xk)
    print("norm of F(x) :", np.linalg.norm(Fxk))
    print("number of iterations : ", iter)











    


