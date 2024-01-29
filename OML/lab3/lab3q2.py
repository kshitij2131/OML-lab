#kshitij jaiswal
#b20cs028

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

R = 6.1 #R=8 is out of range of area and gives impractical values..
lb = 2
ub = 7

def openCSV():
    path = '2_col.xlsx'
    dt = pd.read_excel(path)
    return dt

if __name__ == '__main__':
    dt = openCSV()
    
    B=dt.values
    x,y=B[:,0],B[:,1]

    numPnts = len(x)


    A=np.column_stack((x,np.ones((len(x),1),dtype=float)))

    
    for n in range(lb, ub+1):
        A = np.column_stack((x**n, A))
        beta=np.dot(np.linalg.inv(np.dot(A.T,A)),np.dot(A.T,y.T))
      
        Y = beta[0]*(x**n)
        cost = beta[0]*(R**n)
      
        for i in range(1, n+1):
            Y += (beta[i]*(x**(n-i)))
            cost += (beta[i]*(R**(n-i)))

        print("cost of " + str(R) + "-thousand sq ft house " + str(n) + " poly fitting ", cost)
        print("\n")

        plt.plot(x, y, 'b.')
        plt.plot(x, Y, 'r')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
    

