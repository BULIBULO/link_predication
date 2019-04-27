import numpy as np

#Common Neighbour 

def CN(matrix):
    return np.dot(matrix,matrix)


def Jaccard(Train_MatrixAdjacency):
    Degree_each_Node = sum(Train_MatrixAdjacency)
    CNmatrix = np.dot(Train_MatrixAdjacency,Train_MatrixAdjacency)
    Degree_each_Node.reshape((Degree_each_Node.shape[0],1))
    Degree_each_NodeT = Degree_each_Node.T
    DegreeSum = Degree_each_Node + Degree_each_NodeT
    JCmatrix = CNmatrix / (DegreeSum - CNmatrix)
    JCmatrix = np.nan_to_num(JCmatrix)
    return JCmatrix
     
def RA(Train_MatrixAdjacency):
    Degree_each_Node = sum(Train_MatrixAdjacency)
    Degree_each_Node.reshape((Degree_each_Node.shape[0],1))
    _temp = Train_MatrixAdjacency / Degree_each_Node
    _temp = np.nan_to_num(_temp)

    RAmatrix = np.dot(Train_MatrixAdjacency,_temp)

    return RAmatrix

def AA(Train_MatrixAdjacency):
    Degree_each_Node = np.log(sum(Train_MatrixAdjacency))
    _temp1 = np.nan_to_num(Degree_each_Node)
    _temp1.reshape((Degree_each_Node.shape[0],1))
    _temp2 = Train_MatrixAdjacency / _temp1
    _temp2 = np.nan_to_num(_temp2)

    AA = np.dot(Train_MatrixAdjacency,_temp2)
    return AA

def PA(Train_MatrixAdjacency):
    Degree_each_Node = sum(Train_MatrixAdjacency)
    Degree_each_Node.reshape((Degree_each_Node.shape[0],1))
    Degree_each_NodeT = Degree_each_Node.T
    S = np.dot(Degree_each_Node,Degree_each_NodeT)
    return S

def Katz(Train_MatrixAdjacency,parameter = 0.01):
    I = np.eye(Train_MatrixAdjacency.shape[0])
    S = np.linalg.inv((I - parameter * Train_MatrixAdjacency) ) - I
    return S
