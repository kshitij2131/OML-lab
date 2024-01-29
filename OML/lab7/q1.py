import numpy as np
import matplotlib.pyplot as plt

r = 8
max_iter = 500
numParams = 2
alpha = 0.2
pos_inf = 1e18

def f1(params):
    return (params[0][0] - 2)**2 + (params[1][0] + 2)**2

def f2(params):
    return (params[0][0])**2 + 8*params[1][0]

def fun(params):
    return max(f1(params), f2(params))

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


def subgradFx(params):
    if f1(params) == fun(params):
        return gradFx(params, f1)
    if f2(params) == fun(params):
        return gradFx(params, f2)



if __name__ == '__main__':
    params = np.zeros((numParams, 1))
    alpha = 1/5
    xk = params

    iter = 0
    fbest_val = []
    fbest = pos_inf

    while iter < max_iter:
        dk = -subgradFx(xk)
        xk = xk + alpha*dk
        fcand = fun(xk)

        if fcand < fbest:
            fbest = fcand

        iter += 1
        fbest_val.append(fbest)

    print("optimal value obtained at : ", xk)
    print("optimum value : ", fun(xk))

    plt.plot(range(1, iter+1), fbest_val)
    plt.xlabel('num_iteration')
    plt.ylabel('fbest values')
    plt.title('Subgradient Descent Method')
    plt.show()