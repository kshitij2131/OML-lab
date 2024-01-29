#kshitij jaiswal
#b20cs028

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

R = 8

def openCSV():
    path = "6 columns - Copy.csv"
    dt = pd.read_csv(path)
    return dt

if __name__ == '__main__':
    dt = openCSV()
    
    B=dt.values
    x,y=B[:,0],B[:,1]
    print(y.shape)
    A=np.column_stack((x,np.ones((len(x),1),dtype=float)))
    
    beta=np.dot(np.linalg.inv(np.dot(A.T,A)),np.dot(A.T,y))
    
    print("Cost of " + str(R) + "-thousand sq ft area house ",beta[0]*R + beta[1])
    plt.plot(x, y, 'b.')
    plt.plot(x, beta[0]*x+beta[1], 'r')
    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.show()
    



