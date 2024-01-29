#kshitij jaiswal
#b20cs028

import numpy as np 
from cvxopt import solvers, matrix

r = 8
pair = [(2,2), (3,3), (3,10)]
pi = np.pi
lb=0.0
ub=1.0

    

def fun(params, n, m):
    
    f = []
    gx = 0.0
    for i in range(m, n+1):
        gx += ((params[i-1]-0.5)**2)

    for i in range(m):
        
        prod = 1.0
        if i==0:
            for j in range (1, m):
                prod *= (np.cos(0.5*pi*params[j-1]))
        else:
            for j in range (1, m-i+1):
                prod *= ((np.cos(0.5*pi*params[j-1]))*(np.sin(0.5*pi*params[m-j])))
            
        prod *= (1+gx)
        f.append(prod)

    return f


def gradNhessFun(params, m):
    #x_i = x0
    grad=[]
    hess=[]
    numParams = len(params)
    h = 1e-5
    
    F = fun(params, numParams, m)
    funGrad1 = []
    funGrad2 = []
    for i in range(numParams):
        params[i] += h
        f = fun(params, numParams, m)
        funGrad1.append(f)
        params[i] -= h

    for i in range(numParams):
        params[i] += h
        aux = []
        for j in range(numParams):
            params[j] += h
            f = fun(params, numParams, m)
            aux.append(f)
            params[j] -= h
        funGrad2.append(aux)
        params[i] -= h
    
    for i in range(m):

        g = np.zeros(numParams)

        for j in range(numParams):
            g[j] = (funGrad1[j][i] - F[i])/h
        
        H = np.matrix(np.zeros((numParams, numParams)))
    
        for j in range(numParams):
            for k in range(i+1):
                H[j,k] = (funGrad2[j][k][i] - funGrad1[j][i] - funGrad1[k][i] + F[i])/(h**2)
                H[k,j] = H[j,k]
        grad.append(g)
        hess.append(H)


    return (grad,hess)

    

if __name__ == '__main__':

    grad = []
    hess = []
    for (m,n) in pair:
        params = []
        for i in range(n):
            params.append(np.random.uniform(lb, ub))

        grad, hess = gradNhessFun(params, m)
        params = np.array(params)

        A = np.vstack((-np.identity(n), np.identity(n)))
        b = np.append(params-lb, ub-params)
        
        for i in range(m):
            g = grad[i]
            H = hess[i]

            lmbda = min(np.linalg.eig(H)[0])

            if lmbda > 0.01:
                lmbda = 0
            else:
                lmbda = 0.01-lmbda
            for j in range(n):
                H[j,j] += lmbda

            sol = solvers.qp(matrix(H, tc = 'd'), matrix(g), matrix(A), matrix(b))
            print("Optimal soln for number of parameters " + str(n) + " and f" + str(i+1) +  " -- \n", np.array(sol['x']))
            print('\n')

       



