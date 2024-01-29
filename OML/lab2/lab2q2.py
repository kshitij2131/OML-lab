#kshitij jaiswal
#b20cs028

import numpy as np 
from cvxopt import solvers, matrix

r = 8
nVal = [5,10,30]
pi = np.pi
lb=-1.0
ub=1.0

def I(n):
    I1 = []
    for i in range(2, n+1):
        if (i&1) > 0:
            I1.append(i)

    I2 = []
    for i in range(2, n+1):
        if (i&1) == 0:
            I2.append(i)
    
    return (I1, I2)

def fun1(params, n):
    I1, I2 = I(n)

    x1 = params[0]

    t = 0.0
    for i in I1:
        t += ((params[i-1] - np.sin(6*pi*x1 + i*pi/n))**2)

    t *= (2/np.linalg.norm(I1))
    t += x1

    return t

def fun2(params, n):
    I1, I2 = I(n)

    x1 = params[0]

    t = 0.0
    for i in I2:
        t += ((params[i-1] - np.sin(6*pi*x1 + i*pi/n))**2)

    t *= (2/np.linalg.norm(I2))
    t -= np.sqrt(x1)
    t += 1

    return t

def gradNhessFun1(params):
    #x_i = x0
    g=[]
    numParams = len(params)
    
    F = fun1(params, numParams)
    h, g = 1e-5, np.zeros(numParams)
    fun = []

    for i in range(numParams):
        params[i] += h
        f = fun1(params, numParams)
        g[i] = (f - F)/h
        fun.append(f)
        params[i] -= h
    
    H = np.matrix(np.zeros((numParams, numParams)))
   
    for i in range(numParams):
        params[i] += h
        for j in range(i+1):
            params[j] += h
            H[i,j] = (fun1(params, numParams) - fun[i] - fun[j] + F)/(h**2)
            H[j,i] = H[i,j]
            params[j] -= h
        params[i] -= h

    return (g,H)

def gradNhessFun2(params):
    
    g=[]
    numParams = len(params)
    
    F = fun2(params, numParams)
    h, g = 1e-5, np.zeros(numParams)
    fun = []

    for i in range(numParams):
        params[i] += h
        f = fun2(params, numParams)
        g[i] = (f - F)/h
        fun.append(f)
        params[i] -= h
    
    H = np.matrix(np.zeros((numParams, numParams)))
   
    for i in range(numParams):
        params[i] += h
        for j in range(i+1):
            params[j] += h
            H[i,j] = (fun2(params, numParams) - fun[i] - fun[j] + F)/(h**2)
            H[j,i] = H[i,j]
            params[j] -= h
        params[i] -= h

    return (g,H)
    

if __name__ == '__main__':

   
    for n in nVal:
        params = []
        for i in range(n):
            if i==0:
                params.append(np.random.uniform(0.001, ub))
            else:
                params.append(np.random.uniform(lb, ub))

        g1, H1 = gradNhessFun1(params)
        g2, H2 = gradNhessFun2(params) 
        g = [g1, g2]
        H = [H1, H2]
        params = np.array(params)
        for i in range(2):
            lmbda = min(np.linalg.eig(H[i])[0])

            if lmbda > 0.01:
                lmbda = 0
            else:
                lmbda = 0.01-lmbda
            for j in range(n):
                H[i][j,j] += lmbda

            A = np.vstack((-np.identity(n), np.identity(n)))
            b = np.append(params-lb, ub-params)
            b[0] = params[0] - 0.001

        

            sol = solvers.qp(matrix(H[i], tc = 'd'), matrix(g[i]), matrix(A), matrix(b))
            print("Optimal soln for number of parameters " + str(n) + " and f" + str(i+1) +  " -- \n", np.array(sol['x']))

            print('\n')




