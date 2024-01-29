import numpy as np
import pandas as pd

r = 8.0
numParams = 5
max_iter = 500
epsilon = 1e-5

def openXLSX():
    path = "data set 1_lab.xlsx"
    dt = pd.read_excel(path)
    B=dt.values
    x1,x2,x3,x4,x5,y=B[:,0],B[:,1],B[:,2],B[:,3],B[:,4],B[:,5]

    x1 = x1.astype(float)
    x2 = x2.astype(float)
    x3 = x3.astype(float)
    x4 = x4.astype(float)
    x5 = x5.astype(float)
    y = y.astype(float)
    
    y = np.array(y).reshape(-1, 1)

    numPnts = len(y)
    A=np.column_stack((x1,x2,x3,x4,x5))
    return (A, y)


A, y = openXLSX()

def fun(params):
    return (np.linalg.norm(np.dot(A, params) - y)**2)/2


def grad(params):
    return np.dot(A.T, np.dot(A, params) - y)
    

def prox_op(alphak, x_cap):
    numParams = x_cap.shape[0]
    prox_op_res = np.zeros((numParams, 1))
    for i in range(numParams):
        if x_cap[i] > alphak:
            prox_op_res[i][0] = (x_cap[i][0] - alphak)
        if x_cap[i][0] < -alphak:
            prox_op_res[i][0] = (x_cap[i][0] + alphak)
    return prox_op_res


def comp_opt_prob(x):

    xk = x
    k = 0
    iter = 0

    while iter < max_iter:
        dk = -grad(xk)
        alphak = 1/(1 + np.linalg.norm(dk))
        x_cap = xk + alphak*dk
        # print(x_cap)
        prox = prox_op(alphak, x_cap)
        nrm = np.linalg.norm(xk - prox)
        # print(nrm)
    
        xk = prox
        k += 1
        iter += 1

    return xk




if __name__ == '__main__':
   

    init_params = np.zeros((numParams, 1))
    x_star = comp_opt_prob(init_params)

    print("----SOLUTION FOR 5 COLUMN FITTING USING L1 REGULARIZATION----")
    print("minimum is obtained at x* =  ", x_star)
    print("minimum value of objective function at x* = ", fun(x_star) + np.linalg.norm(x_star, ord=1))
    for i in range(numParams):
        if x_star[i][0] < 0:
            x_star[i][0] = 0
    
    price_pred = np.array([r/10, 10*r, r+50, 2*r, 2])

    print("predicted price at given point = ", np.dot(price_pred, x_star))    

    



