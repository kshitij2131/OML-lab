import numpy as np 
from cvxopt import solvers, matrix

r = 8
lb = -2*r  #-16
ub = 2*r  #16

def fun(params):
    x1 = params[0]
    x2 = params[1]
    return (x1-r)**2 + (x2-r)**2

def gradNhess(params):
    #x_i = x0
    g=[]
    numParams = len(params)
    
    F = fun(params)
    h, g = 1e-5, np.zeros(numParams)
    fun1 = []

    for i in range(numParams):
        params[i] += h
        f1 = fun(params)
        g[i] = (f1 - F)/h
        fun1.append(f1) # f eval at (params[i]+h)
        params[i] -= h
    
    H = np.matrix(np.zeros((numParams, numParams)))
   
    for i in range(numParams):
        params[i] += h
        for j in range(i+1):
            params[j] += h
            H[i,j] = (fun(params) - fun1[i] - fun1[j] + F)/(h**2)
            H[j,i] = H[i,j]
            params[j] -= h
        params[i] -= h

    return (g,H)


if __name__ == '__main__':
    params = np.array([-0.5*r, 1.5*r])
    numParams = len(params)
    gradient, hessian = gradNhess(params)

  

    lmbda = min(np.linalg.eig(hessian)[0])
    if lmbda > 0.01:
        lmbda = 0
    else:
        lmbda = 0.01-lmbda
    for i in range(numParams):
        hessian[i,i] += lmbda

    A = np.vstack((-np.identity(numParams), np.identity(numParams)))
    print(params)
    b = np.append(params-lb, ub-params)
    print(b)

    sol = solvers.qp(matrix(hessian, tc = 'd'), matrix(gradient), matrix(A), matrix(b))
    # print(np.array(sol['x']))
    # print(gradient)
    # print(hessian)




