import numpy as np
from numpy import linalg


r = 8.00
delta = 0.01
maxIter = 500
epsilon = 1e-4


def fun(params):
    x1 = params[0][0]
    x2 = params[1][0]

    return ((x1 - r)**4) + ((x1 - 22*x2)**2)

def gradFun1(params):
    x1 = params[0][0]
    x2 = params[1][0]

    return 4*((x1 - r)**3) + 2*(x1 - 22*x2)

def gradFun2(params):
    x1 = params[0][0]
    x2 = params[1][0]

    return (-44)*(x1 - 22*x2)


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
    gradFx1 = gradFx(params, gradFun1)
    gradFx2 = gradFx(params, gradFun2)

    j = np.vstack((gradFx1.T, gradFx2.T))
    return j




if __name__ == '__main__':

    params = np.array([[r-1], [r+1]])
    numParams = params.shape[0]


    #using newton's method..
    
    iter = 0
    xk = params
    gradFxk = np.vstack((gradFun1(xk), gradFun2(xk)))
    jacGradFxk = jacobianFx(xk)

    gradEvals = 2*(1+numParams)
    funcEvals = 0

    
    while np.linalg.norm(gradFxk) >= epsilon and iter < maxIter:
        dk = -np.linalg.solve(jacGradFxk, gradFxk)
        xk = xk + dk
        gradFxk = np.vstack((gradFun1(xk), gradFun2(xk)))
        jacGradFxk = jacobianFx(xk)
        gradEvals += 2*(1+numParams)
        iter += 1

    print("optimal solution for x (using newton's method): ", xk)
    print("norm of grad(f(x)) :", np.linalg.norm(gradFxk))
    print("number of function calls :", funcEvals)
    print("number of gradient of function calls :", gradEvals)
    print("number of iterations : ", iter)











    


