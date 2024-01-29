#kshitij jaiswal
#b20cs028

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

R = 8


def openCSV():
    path = "Census data (Chandigarh).csv"
    dt = pd.read_csv(path)
    return dt

if __name__ == '__main__':
    dt = openCSV()
    
    B=dt.values
    x,y=B[:,0],B[:,1]

    print(x)

    yp = np.log(y)
    print(yp)

    A=np.column_stack((x,np.ones((len(x),1),dtype=float)))
    beta=np.dot(np.linalg.inv(np.dot(A.T,A)),np.dot(A.T,yp.T))
    
    pop = (np.e)**(beta[0]*2021 + beta[1])
    print("Population of Chandigarh in 2021 ", int(pop))
    plt.plot(x, y, 'b.')
    Y = (np.e)**(beta[0]*x+beta[1])
    plt.plot(x, Y, 'r')
    plt.xlabel('x')
    plt.ylabel('y')
  
    plt.show()
    

