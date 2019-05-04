import numpy as np


def cn(train_matrix):
    return np.dot(train_matrix, train_matrix)


def jc(train_matrix):
    nodes_degree = sum(train_matrix)
    cn_matrix = np.dot(train_matrix, train_matrix)
    nodes_degree.reshape((nodes_degree.shape[0], 1))
    nodes_degree_transposed = nodes_degree.T
    degree_sum = nodes_degree + nodes_degree_transposed
    jc_matrix = cn_matrix / (degree_sum - cn_matrix)
    jc_matrix = np.nan_to_num(jc_matrix)
    return jc_matrix


def ra(train_matrix):
    nodes_degree = sum(train_matrix)
    nodes_degree.reshape((nodes_degree.shape[0], 1))
    _temp = train_matrix / nodes_degree
    _temp = np.nan_to_num(_temp)
    ra_matrix = np.dot(train_matrix, _temp)
    return ra_matrix


def aa(train_matrix):
    nodes_degree = np.log(sum(train_matrix))
    _temp1 = np.nan_to_num(nodes_degree)
    _temp1.reshape((nodes_degree.shape[0], 1))
    _temp2 = train_matrix / _temp1
    _temp2 = np.nan_to_num(_temp2)
    aa_matrix = np.dot(train_matrix, _temp2)
    return aa_matrix


def pa(train_matrix):
    nodes_degree = sum(train_matrix)
    nodes_degree.reshape((nodes_degree.shape[0], 1))
    nodes_degree_transposed = nodes_degree.T
    pa_matrix = np.dot(nodes_degree, nodes_degree_transposed)
    return pa_matrix


def katz(train_matrix, parameter=0.01):
    I = np.eye(train_matrix.shape[0])
    S = np.linalg.inv((I - parameter * train_matrix)) - I
    return S
