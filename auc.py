import numpy as np


def calculating_auc(train_matrix, test_matrix, similarity_matrix, max_node):
    auc_num = 60000
    similarity_matrix = np.triu(similarity_matrix - similarity_matrix * train_matrix)
    matrix_not_exist = np.ones(max_node) - train_matrix - test_matrix - np.eye(max_node)
    test = np.triu(test_matrix)
    not_exist = np.triu(matrix_not_exist)
    test_num = len(np.argwhere(test == 1))
    not_exist_num = len(np.argwhere(not_exist == 1))
    test_random = [int(x) for index, x in enumerate((test_num * np.random.rand(1, auc_num))[0])]
    not_exist_random = [int(x) for index, x in enumerate((not_exist_num * np.random.rand(1, auc_num))[0])]
    test_prediction = similarity_matrix * test
    not_exist_prediction = similarity_matrix * not_exist

    test_index = np.argwhere(test == 1)
    test_data = np.array([test_prediction[x[0], x[1]] for index, x in enumerate(test_index)]).T

    not_exist_index = np.argwhere(not_exist == 1)
    not_exist_data = np.array([not_exist_prediction[x[0], x[1]] for index, x in enumerate(not_exist_index)]).T

    test_random = np.array([test_data[x] for index, x in enumerate(test_random)])
    not_exist_random = np.array([not_exist_data[x] for index, x in enumerate(not_exist_random)])
    n = 0
    for _ in range(auc_num):
        if test_random[_] > not_exist_random[_]:
            n += 1
        elif test_random[_] == not_exist_random[_]:
            n += 0.5
        else:
            n += 0
    auc = n / auc_num
    return auc
