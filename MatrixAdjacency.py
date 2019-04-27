import numpy as np


def TransfromMatrix(Data,MaxNode):
    MatrixAdjacency = np.zeros((int(MaxNode), int(MaxNode)), dtype=np.int8)
    for i in range(Data.shape[0]):
        x = int(Data[i][0])
        y = int(Data[i][1])
        MatrixAdjacency[x, y] = 1
        MatrixAdjacency[y, x] = 1
    return MatrixAdjacency
