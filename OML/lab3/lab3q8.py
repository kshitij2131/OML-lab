#kshitij jaiswal
#b20cs028

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


lb = 1
ub = 20

def polynomial_fit(x, y, p):

    numPnts = len(x)
    A = np.ones((numPnts,1),dtype=float)
    for i in range(1, p+1):
        A = np.column_stack((x**i, A))

    beta=np.dot(np.linalg.inv(np.dot(A.T,A)),np.dot(A.T,y.T))

    Y = beta[0]*(x**p)
    for i in range(1, p+1):
        Y += (beta[i]*(x**(p-i)))
    
    # plt.plot(x, y, 'b.')
    # plt.plot(x, Y, 'r')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.show()
    
    loss = 0.0
    for i in range(numPnts):
        loss += np.abs(Y[i] - y[i])
    loss /= numPnts

    return loss

def openCSV():
    path = '2_col.xlsx'
    dt = pd.read_excel(path)
    return dt

if __name__ == '__main__':

    dt = openCSV()
    B=dt.values

    x,y=B[:,0],B[:,1]

    avgLoss = []
    minLoss = 1e18
    bestDeg = 0
    for p in range(lb, ub+1):
        lossVal = polynomial_fit(x,y,p)
        avgLoss.append(lossVal)
        if lossVal < minLoss:
            minLoss = lossVal
            bestDeg = p
    
    print("minimum average loss is " + str(minLoss) + " which is achieved at degree " + str(bestDeg))

    plt.plot(list(range(lb, ub+1)), avgLoss, 'r')
    plt.xlabel('polynomial degree')
    plt.ylabel('average absolute loss')
    plt.show()
    

