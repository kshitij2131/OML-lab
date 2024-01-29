import numpy as np
from numpy import linalg


r = 8.00
delta = 0.01
maxIter = 500
alpha = 1.00
r0 = 0.50  
beta1 = 1e-4
beta2 = 0.9
epsilon = 1e-3
gradEvals = 0
funcEvals = 0
alpha_lower = 1e-5
numParams = 10
Q = np.array([[5.9869,1.9195,0.6808,1.1705,0.4476,1.5025,0.5102,1.0119,1.3982,1.7818],
            [1.9195,9.6743,1.0944,0.2772,0.2986,0.515,1.6814,0.5086,1.6286,0.487],
            [0.6808,1.0944,9.4341,0.7,0.3932,0.5022,1.2321,0.9466,0.7033,1.6617],
            [1.1705,0.2772,0.7,6.6821,1.0994,1.8344,0.5717,1.5144,1.5075,0.7609],
            [0.4476,0.2986,0.3932,1.0994,6.5426,0.1517,0.1079,1.0616,1.5583,1.868],
            [1.5025,0.515,0.5022,1.8344,0.1517,3.0392,1.1376,0.9388,0.0238,0.6742],
            [0.5102,1.6814,1.2321,0.5717,0.1079,1.1376,3.2975,1.5886,0.6224,1.0571],
            [1.0119,0.5086,0.9466,1.5144,1.0616,0.9388,1.5886,3.3252,1.204,0.5259],
            [1.3982,1.6286,0.7033,1.5075,1.5583,0.0238,0.6224,1.204,7.2326,1.3784],
            [1.7818,0.487,1.6617,0.7609,1.868,0.6742,1.0571,0.5259,1.3784,7.9852]])


def fun(params):
    global funcEvals 
    funcEvals += 1

    funval = 0.0
    for i in range(numParams-1):
        funval += (100*((params[i+1][0] - params[i][0]**2)**2) + (1 - params[i][0]**2))

    return funval

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

    params = np.zeros((numParams,1))
    for i in range(numParams):
        params[i][0] = 0.5


    #using inexact line search algorithm along with bergman's distance..
    
    iter = 0
    xk = params
    gradFxk = gradFx(xk, fun)

    
    
    while linalg.norm(gradFxk) >= epsilon and iter < maxIter:
        dk = -1 * np.linalg.solve(Q, gradFxk)
    
        alpha = 1.00
        while (armijo(xk, alpha, dk, gradFxk) == False or wolfe(xk, alpha, dk, gradFxk) == False) and alpha >= alpha_lower:
            alpha = alpha*r0

        xk = xk + alpha*dk
        gradFxk = gradFx(xk, fun)
        iter += 1

    print("optimal solution for x (using mirror descent): ", xk)
    print("norm of grad(f(x)) :", np.linalg.norm(gradFxk))
    print("optimum value of F(x_k) :", fun(xk))
    print("number of function calls :", funcEvals)
    print("number of gradient of function calls :", gradEvals)
    print("number of iterations : ", iter)







    


