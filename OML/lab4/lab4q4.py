import numpy as np
from cvxopt import matrix, solvers


class node:

    def __init__(self, val, edges) -> None:
        self.val = val
        self.incoming = []
        self.outgoing = []
        for (u, v) in edges:
            if self.val == u:
                self.outgoing.append(v)
            if self.val == v:
                self.incoming.append(u)

    
        
if __name__ == '__main__':

    numNodes = 10
    numEdges = 20

    start = 1
    end = numNodes

    edges = [(1, 2), (1, 3), (1, 4), (2, 5), (2, 6), (2, 7), (3, 5), (3, 6), (3, 7), (4, 5), (4, 6), (4, 7), (5, 8), (5, 9), (6, 8), (6, 9), (7, 8), (7, 9), (8, 10), (9, 10)]
    # cost = {}


    lstNodes = []

    for i in range(1, numNodes+1):
        x = node(i, edges)
        lstNodes.append(x)

    A = np.zeros(((numNodes + numEdges)<<1, numEdges))

    for i in range(0, numNodes<<1, 2):
        obj = lstNodes[i>>1]
        val = obj.val
        for out in obj.outgoing:
            A[i, edges.index((val, out))] = 1
            A[i+1, edges.index((val, out))] = -1
        for inc in obj.incoming:
            A[i, edges.index((inc, val))] = -1
            A[i+1, edges.index((inc, val))] = 1
    
    for i in range(numNodes<<1, (numNodes<<1)+numEdges):
        A[i, i-(numNodes<<1)] = -1
        A[i+numEdges, i-(numNodes<<1)] = 1
    
    # print(A)

    b = np.zeros(((numNodes + numEdges)<<1, 1))
    for i in range(0, numNodes<<1, 2):
        nodeNum = (i>>1) + 1
        if nodeNum == start:
            b[i] = 1
            b[i+1] = -1
        elif nodeNum == end:
            b[i] = -1
            b[i+1] = 1
        else:
            b[i] = 0
            b[i+1] = 0
    
    for i in range((numNodes<<1)+numEdges, (numNodes + numEdges)<<1):
        b[i] = 1
    
    # print(b)

    c = np.array([[4], [6], [6], [6], [8], [9], [5], [4], [6], [5], [5], [7], [6], [8], [4], [9], [3], [7], [9], [6]])

    soln = solvers.lp(matrix(c, tc='d'), matrix(A, tc='d'), matrix(b, tc='d'))

    print(np.round_(soln['x'], decimals=5), soln['primal objective'])  

        


    
