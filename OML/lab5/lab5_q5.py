import numpy as np
from numpy import linalg
from cvxopt import solvers, matrix

r = 8.00
delta = 0.01
maxIter = 500
epsilon = 1e-4
pi = np.pi
n = 30

def I():
    I1 = []
    for i in range(2, n+1):
        if (i&1) > 0:
            I1.append(i)

    I2 = []
    for i in range(2, n+1):
        if (i&1) == 0:
            I2.append(i)
    
    return (I1, I2)

I1, I2 = I()

def fun1(params):
    x1 = params[0][0]

    t = 0.0
    for i in I1:
        t += ((params[i-1][0] - np.sin(6*pi*x1 + i*pi/n))**2)

    t *= (2/linalg.norm(I1))
    t += x1

    return t

def fun2(params):
    x1 = params[0][0]

    t = 0.0
    for i in I2:
        t += ((params[i-1][0] - np.sin(6*pi*x1 + i*pi/n))**2)

    t *= (2/linalg.norm(I2))
    t -= np.sqrt(x1)
    t += 1

    return t

def fun(params):
    return (r/10)*fun1(params) + (1-(r/10))*fun2(params)


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

def hessianFx(params, fx):
    h = 1e-5
    numParams = params.shape[0]
    H = np.matrix(np.zeros((numParams, numParams)))
    F = fx(params)
   
    for i in range(numParams):
        params[i][0] += h
        fxi = fx(params)

        for j in range(i+1):
            params[j][0] += h

            params[i][0] -= h
            fxj = fx(params)
            params[i][0] += h

            H[i,j] = (fx(params) - fxi - fxj + F)/(h**2)
            H[j,i] = H[i,j]
            params[j][0] -= h
        params[i][0] -= h
    
    
    #adding the regularization parameter..
    lmbda = min(linalg.eig(H)[0])
    if lmbda > delta:
        lmbda = 0
    else:
        lmbda = delta-lmbda
    for i in range(numParams):
        H[i,i] += lmbda
    
    

    return H




if __name__ == '__main__':

    params = np.zeros((n, 1))
    numParams = params.shape[0]
    for i in range(numParams):
        if i == 0:
            params[i][0] = np.random.uniform(0.001, 1.0)
        else:
            params[i][0] = np.random.uniform(-1.0, 1.0)


    #using modified newton's method..
    
    iter = 0
    xk = params
    gradFxk = gradFx(xk, fun)
    hessFxk = hessianFx(xk, fun)
    A = np.vstack((-np.identity(numParams), np.identity(numParams)))
    
    while linalg.norm(gradFxk) >= epsilon and iter < maxIter:
        b = np.append(xk-(-1.0), 1.0-xk)
        b[0] += (-1.0)
        b[0] -= 0.001
    
        sol = solvers.qp(matrix(hessFxk, tc = 'd'), matrix(gradFxk), matrix(A), matrix(b))
        dk = np.array(sol['x'])
        xk = xk + dk
                    
        gradFxk = gradFx(xk, fun)
        hessFxk = hessianFx(xk, fun)
        iter += 1

    print("optimal solution for x (using modified newton's method): ", xk)
    print("minimum value of F(x) :", fun(xk))
    print("norm of grad(F(x)) :", linalg.norm(gradFxk))
    print("number of iterations : ", iter)











    


