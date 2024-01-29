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
    x,y,z=B[:,1],B[:,2],B[:,0]

    numPnts = len(x)
    
    A = np.ones((numPnts,1),dtype=float)
    A = np.column_stack((y, A))
    A = np.column_stack((x, A))
    A = np.column_stack((y**2, A))
    A = np.column_stack((x*y, A))
    A = np.column_stack((x**2, A))

    beta=np.dot(np.linalg.inv(np.dot(A.T,A)),np.dot(A.T,z.T))
    price = beta[0]*(R**2) + beta[1]*(R*(R+3)) + beta[2]*((R+3)**2) + beta[3]*(R) + beta[4]*(R+3) + beta[5]

    print("cost of " + str(R) + "-thousand sq ft house & " + str(R+3) + " bedrooms ", price)
    print("\n")
    

