#kshitij jaiswal
#b20cs028

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

R = 8
lmbda = abs(R/10 - 5)

def openCSV():
    path = "6 columns.csv"
    dt = pd.read_csv(path)
    return dt

if __name__ == '__main__':
    dt = openCSV()
    numParams = 4
    
    B=dt.values
    x1,x2,x3,y=B[:,0],B[:,1],B[:,2], B[:,5]

    numPnts = len(x1)
    
    A = np.ones((numPnts,1),dtype=float)
    A = np.column_stack((x3, A))
    A = np.column_stack((x2, A))
    A = np.column_stack((x1, A))

    lhs = np.dot(A.T, A) + ((lmbda)*np.identity(numParams))
    rhs = np.dot(A.T, y)
    
    beta=np.dot(np.linalg.inv(lhs),rhs)

    print("coefficient of area in best fitting hyperplane", beta[0])
    print("coefficient of bedrooms in best fitting hyperplane", beta[1])
    print("coefficient of bathrooms in best fitting hyperplane", beta[2])
    print("constant coefficient (β) in best fitting hyperplane", beta[1])

    print("\n")
    

