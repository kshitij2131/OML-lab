#kshitij jaiswal
#b20cs028

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

R = 8
lmbda = abs(R/10 - 5)

def openXLSX():
    path = "2_col_revised.xlsx"
    dt = pd.read_excel(path)
    return dt

if __name__ == '__main__':
    dt = openXLSX()
    numParams = 2
    
    B=dt.values
    x,y=B[:,0],B[:,1]
    x[-1] = R
    y[-1] = 2*R + 3.5
    x = x.astype(float)
    y = y.astype(float)

    A=np.column_stack((x,np.ones((len(x),1),dtype=float)))
    lhs = np.dot(A.T, A) + ((lmbda)*np.identity(numParams))
    rhs = np.dot(A.T, y)
    
    beta=np.dot(np.linalg.inv(lhs),rhs)

    print("minmum f is obtained at x* = ", beta)
    
    plt.plot(x, y, 'b.')
    plt.plot(x, beta[0]*x+beta[1], 'r')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    



