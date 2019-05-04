import numpy as np


def transform_matrix(data, max_node):
    matrix_adjacency = np.zeros((int(max_node), int(max_node)), dtype=np.int8)
    for i in range(data.shape[0]):
        x = int(data[i][0])
        y = int(data[i][1])
        matrix_adjacency[x, y] = 1
        matrix_adjacency[y, x] = 1
    return matrix_adjacency
