#kshitij jaiswal
#b20cs028

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

r = 8.0
lmbda = 0.5
# lmbda = 1
numParams1 = 2
numParams2 = 4
max_iter = 1000
epsilon = 1e-5

def openXLSX():
    path = "2_col_revised.xlsx"
    dt = pd.read_excel(path)
    B=dt.values
    x,y=B[:,0],B[:,1]
    
    x[-1] = r
    y[-1] = 2*r + 3.5

    x = x.astype(float)
    y = y.astype(float)
    
    y = np.array(y).reshape(-1, 1)

    numPnts = len(x)
    A=np.column_stack((x,np.ones((numPnts,1),dtype=float)))
    return (A, y, x)

def openCSV():
    path = "6 columns.csv"
    dt = pd.read_csv(path)
    B=dt.values
    x1,x2,x3,y=B[:,0],B[:,1],B[:,2], B[:,5]
    y = np.array(y).reshape(-1, 1)

    numPnts = len(x1)
    A = np.ones((numPnts,1),dtype=float)
    A = np.column_stack((x3, A))
    A = np.column_stack((x2, A))
    A = np.column_stack((x1, A))
    return (A, y)

A1, y1, x1 = openXLSX()
A2, y2 = openCSV()

def fun1(params):
    return (np.linalg.norm(np.dot(A1, params) - y1)**2)/(2*lmbda)

def fun2(params):
    return (np.linalg.norm(np.dot(A2, params) - y2)**2)/(2*lmbda)

def grad(params, fun):
    if fun == fun1:
        return (1/lmbda)*np.dot(A1.T, np.dot(A1, params) - y1)
    if fun == fun2:
        return (1/lmbda)*np.dot(A2.T, np.dot(A2, params) - y2)
    return -1

def prox_op(alphak, x_cap):
    numParams = x_cap.shape[0]
    prox_op_res = np.zeros((numParams, 1))
    for i in range(numParams):
        if x_cap[i] > alphak:
            prox_op_res[i][0] = (x_cap[i][0] - alphak)
        if x_cap[i][0] < -alphak:
            prox_op_res[i][0] = (x_cap[i][0] + alphak)
    return prox_op_res


def comp_opt_prob(x, fun):

    xk = x
    k = 0
    iter = 0

    while iter < max_iter:
        alphak = 1/(r + np.linalg.norm(grad(xk, fun)))
        x_cap = xk - alphak*grad(xk, fun)
        # print(x_cap)
        prox = prox_op(alphak, x_cap)
        nrm = np.linalg.norm(xk - prox)
        # print(nrm)
    
        xk = prox
        k += 1
        iter += 1

    return xk




if __name__ == '__main__':
   

    init_params1 = np.ones((numParams1, 1))
    x_star = comp_opt_prob(init_params1, fun1)

    print("----SOLUTION FOR 2 COLUMN FITTING USING L1 REGULARIZATION----")
    print("minimum is obtained at x* =  ", x_star)
    print("minimum value of objective function at x* = ", lmbda*fun1(x_star) + lmbda*np.linalg.norm(x_star, ord=1))
    
    plt.plot(x1, y1, 'b.')
    plt.plot(x1, x_star[0]*x1 + x_star[1], 'r')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    init_params2 = np.ones((numParams2, 1))
    x_star = comp_opt_prob(init_params2, fun2)

    print("\n")
    print("----SOLUTION FOR 4 COLUMN FITTING USING L1 REGULARIZATION----")
    print("minimum is obtained at x* =  ", x_star)
    print("minimum value of objective function at x* = ", lmbda*fun2(x_star) + lmbda*np.linalg.norm(x_star, ord=1))

    



