#kshitij jaiswal
#b20cs028

import numpy as np
import pandas as pd

r = 8.0
numParams = 2
max_iter = 500
epsilon = 1e-5



def fun(params):
    x1 = params[0][0]
    x2 = params[1][0]
    return ((x1-2)**2) + ((x2-2)**2)

def gradFx(params):
    numParams = params.shape[0]
    F = fun(params)
    h, g = 1e-5, np.zeros((numParams, 1))

    for i in range(numParams):
        params[i][0] += h
        f = fun(params)
        g[i][0] = (f - F)/h
        params[i][0] -= h
    
    return g


def prox_op_l1(alphak, x_cap):
    prox_op_res = np.zeros((numParams, 1))
    for i in range(numParams):
        if x_cap[i][0] > alphak:
            prox_op_res[i][0] = (x_cap[i][0] - alphak)
        if x_cap[i][0] < -alphak:
            prox_op_res[i][0] = (x_cap[i][0] + alphak)
    return prox_op_res

def comp_opt_prob(x, alpha):
    xk = x
    alphak = alpha 

    k = 0
    iter = 0
    prox_op = prox_op_l1(alphak, xk - alphak*gradFx(xk))

    while iter < max_iter and np.linalg.norm(xk - prox_op) >= epsilon:
        xk = prox_op
        k += 1
        alphak = 1/(r+k)
        prox_op = prox_op_l1(alphak, xk - alphak*gradFx(xk))
        iter += 1

    return (xk, iter)




if __name__ == '__main__':
    init_params = np.zeros((numParams, 1))
    alpha = 1/r
    x_star, iter = comp_opt_prob(init_params, alpha)

    print("number of iterations = ", iter)
    print("minimum is obtained at x* =  ", x_star)
    print("minimum value of f + ||x|| at x* = ", 0.5*fun(x_star) + 0.5*np.linalg.norm(x_star, ord=1))

    



