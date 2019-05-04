from auc import calculating_auc
from algorithm import *
from matrix_adjacency import transform_matrix

dataset = 'USAir'
# dataset = ['Grid', 'PB', 'USAir']
training_set = np.loadtxt('dataset/' + dataset + '_train.txt')
test_set = np.loadtxt('dataset/' + dataset + '_test.txt')
max_node = int(training_set.max() + 1)
train_matrix_adjacency = transform_matrix(training_set, max_node)
test_matrix_adjacency = transform_matrix(test_set, max_node)

algorithms = [cn, jc, ra, aa, pa, katz]
similarity_matrix = []
auc_result = {}
for i in algorithms:
    similarity_matrix.append(i(train_matrix_adjacency))
    temp = i(train_matrix_adjacency)
    auc_result[i.__name__] = calculating_auc(train_matrix_adjacency, test_matrix_adjacency, temp, max_node)

print(auc_result)
