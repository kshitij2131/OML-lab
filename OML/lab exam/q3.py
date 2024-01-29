import numpy as np
from cvxopt import matrix, solvers
from cvxopt.solvers import qp
import pandas as pd

r = 28.0
a = 22.0

class svm:
    def trainPrimal(self,X,y):
        v = X.shape[0] + X.shape[1] + 1
        I1, I2 = np.identity(v), np.ones(v)
        I1[X.shape[1]:,:] = 0.0
        I2[:X.shape[1]+1] = 0.0

        q = matrix(I1, tc='d')
        p = matrix(I2, tc='d')

        e = -np.identity(X.shape[0])
        e = np.concatenate((e,e), axis=0)

        x = np.column_stack((X, -np.ones(X.shape[0])))*(-y.reshape(-1,1))
        x = np.concatenate((x, np.zeros((X.shape[0], X.shape[1]+1))), axis=0)

        G = matrix(np.concatenate((x,e), axis=1), tc='d')
        h = matrix(np.concatenate((-np.ones(X.shape[0]), np.zeros(X.shape[0])), axis=0), tc='d')

        sol = qp(q, p, G, h)
        fval = sol['primal objective']
        w,b = sol['x'][:X.shape[1]], sol['x'][X.shape[1]]

        return w,b,fval
    
    def trainDual(self,X,y):
        num_datapoints = X.shape[0]
        X = X.T
        Y = np.diag(y)
        q = matrix(np.dot(np.dot(Y, np.dot(X.T, X)), Y), tc='d')
        p = matrix(-np.ones(num_datapoints), tc='d')
        g = matrix(np.concatenate((-np.identity(num_datapoints), np.identity(num_datapoints)), axis=0), tc='d')
        h = matrix(np.concatenate((np.zeros(num_datapoints), np.ones(num_datapoints)), axis = 0), tc='d')
        a = matrix(y.reshape(1,-1), tc='d')
        b = matrix(np.zeros(1), tc='d')
        sold = qp(q, p, g, h, a, b)
        var = sold['x']
        fval = sold['primal objective']
        return var, fval
    
if __name__ == '__main__':
    
    df = pd.read_excel('insulin_2.xlsx')
    df.columns = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'y']
    df['y'] = df['y'].apply(lambda x: -1 if x == 0 else 1)
    df['x1'] = df['x1'].apply(lambda x: float(x))
    df['x2'] = df['x2'].apply(lambda x: float(x))
    df['x3'] = df['x3'].apply(lambda x: float(x))
    df['x4'] = df['x4'].apply(lambda x: float(x))
    df['x5'] = df['x5'].apply(lambda x: float(x))
    df['x6'] = df['x6'].apply(lambda x: float(x))
    df['x7'] = df['x7'].apply(lambda x: float(x))
    df['x8'] = df['x8'].apply(lambda x: float(x))

    X = df[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']].values
    y = df['y'].values


    model = svm()
    print("----------- MODEL DUAL -----------")
    vard, fvald = model.trainDual(X,y)
    print('var =', vard)
    print('fval =', fvald)
    

    print("----------- MODEL PRIMAL -----------")
    w,b,fvalp = model.trainPrimal(X,y)

    print('weights =', w)
    print('bias =', b)
    print('fval =', fvalp)

    pred_pnt = np.array([0, 100+r, 55+r, 25+(r/10), 0, 5*r/8, r/10, a+140])
    pred_val = np.dot(pred_pnt, w)+b
    obs_cutoff = 12.0
    print(pred_val)
    print("predicted outcome at given point : ", 1 if pred_val > obs_cutoff else 0)