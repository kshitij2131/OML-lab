import numpy as np
from numpy import linalg


r = 8.00
delta = 0.01
maxIter = 5000
alpha = 1.00
r0 = 0.50  
beta1 = 1e-4
beta2 = 0.9
epsilon = 1e-4

gradEvals = 0
funcEvals = 0


def fun(params):
    global funcEvals 
    funcEvals += 1

    x1 = params[0][0]
    x2 = params[1][0]

    return ((x1 - r)**4) + ((x1 - 2*x2)**2)

def gradFx(params, fx):
    global gradEvals 
    gradEvals += 1
    numParams = params.shape[0]
    
    F = fx(params)

    h, g = 1e-5, np.zeros(numParams)

    for i in range(numParams):
        params[i][0] += h
        f = fx(params)
        g[i] = (f - F)/h
        params[i][0] -= h
    
    return g.reshape(-1, 1)


def armijo(xk, alphak, dk, gradFxk):
    lhs = fun(xk + alphak*dk)
    rhs = fun(xk) + alphak*beta1*np.dot(gradFxk.T, dk)
    if lhs <= rhs:
        return True
    return False

def wolfe(xk, alphak, dk, gradFxk):
    lhs = np.dot(gradFx(xk + alphak*dk, fun).T, dk)
    rhs = beta2*np.dot(gradFxk.T, dk)
    if lhs >= rhs:
        return True
    return False



if __name__ == '__main__':

    params = np.array([[r-1], [r+1]])
    numParams = params.shape[0]


    #using inexact line search algorithm along with steepest descent..
    
    iter = 0
    xk = params
    gradFxk = gradFx(xk, fun)

    while linalg.norm(gradFxk) >= epsilon and iter < maxIter:
        dk = -1 * gradFxk
    
        alpha = 1.00
        while armijo(xk, alpha, dk, gradFxk) == False or wolfe(xk, alpha, dk, gradFxk) == False:
            alpha = alpha*r0

        xk = xk + alpha*dk
        gradFxk = gradFx(xk, fun)
        iter += 1

    print("optimal solution for x (using steepest descent method): ", xk)
    print("value of F(x_k) :", fun(xk))
    print("norm of grad(f(x)) :", np.linalg.norm(gradFxk))
    print("number of function calls :", funcEvals)
    print("number of gradient of function calls :", gradEvals)
    print("number of iterations : ", iter)







    


