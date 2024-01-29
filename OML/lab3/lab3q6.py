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

def numBath(R):
    if R&1: 
        return 2
    return 1

def numStorey(R):
    return (R%4) + 1

def numAC(R):
    if R&1: 
        return 0
    return 1

if __name__ == '__main__':
    dt = openCSV()
    
    B=dt.values
    x1,x2,x3,x4,x5,x6=B[:,1],B[:,2],B[:,3],B[:,4],B[:,5],B[:,0]

    numPnts = len(x1)
    
    A = np.ones((numPnts,1),dtype=float)
    A = np.column_stack((x4, A))
    A = np.column_stack((x3, A))
    A = np.column_stack((x2, A))
    A = np.column_stack((x1, A))

    beta=np.dot(np.linalg.inv(np.dot(A.T,A)),np.dot(A.T,x6.T))
    price = beta[0]*R + beta[1]*(R+3) + beta[2]*(numBath(R)) + beta[3]*(numStorey(R))

    print("cost of " + str(R) + "-thousand sq ft house, " + str(R+3) + " bedrooms, " + str(numBath(R)) + " bathroom(s) & " + str(numStorey(R)) + " storeys ", price)
    print("\n")
    
    A = np.ones((numPnts,1),dtype=float)
    A = np.column_stack((x5, A))
    A = np.column_stack((x4, A))
    A = np.column_stack((x3, A))
    A = np.column_stack((x2, A))
    A = np.column_stack((x1, A))
    beta=np.dot(np.linalg.inv(np.dot(A.T,A)),np.dot(A.T,x6.T))
    price = beta[0]*R + beta[1]*(R+3) + beta[2]*(numBath(R)) + beta[3]*(numStorey(R)) + beta[4]*(numAC(R))

    print("cost of " + str(R) + "-thousand sq ft house, " + str(R+3) + " bedrooms, " + str(numBath(R)) + " bathroom(s), " + str(numStorey(R)) + " storey(s) & " + str(numAC(R)) + " AC(s) ", price)
    print("\n")
