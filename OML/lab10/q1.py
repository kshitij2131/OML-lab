import numpy as np
from cvxopt import matrix, solvers
from cvxopt.solvers import qp
import pandas as pd

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
    df = pd.read_csv('generated_test.csv', header=None)
    df.columns = ['x1', 'x2', 'target']
    df['target'] = df['target'].apply(lambda x: 1 if x == 1 else -1)
    df['x1'] = df['x1'].apply(lambda x: float(x))
    df['x2'] = df['x2'].apply(lambda x: float(x))

    X = df[['x1', 'x2']].values
    y = df['target'].values
    model = svm()
    var, fvalp = model.trainDual(X,y)
    print('var =', var)
    print('fval =', fvalp)
    
    w,b,fvalp = model.trainPrimal(X,y)

    print('w =', w)
    print('b =', b)
    print('fval =', fvalp)