from auc import calculating_auc
from algorithm import *
from matrix_adjacency import transform_matrix
from matplotlib import pyplot as plt


def draw(dataset=None, result=None):
    # dataset = ['PB', 'USAir']
    # result = {
    #     'PB': {'cn': 0.9311833333333334, 'jc': 0.743475, 'ra': 0.843675, 'aa': 0.88715, 'pa': 0.5, 'katz': 0.9385},
    #     'USAir': {'cn': 0.9623, 'jc': 0.8017333333333333, 'ra': 0.8901666666666667, 'aa': 0.9239, 'pa': 0.5,
    #               'katz': 0.9611}}
    flag = 121
    plt.figure(figsize=(12, 4))
    for _ in dataset:
        keys = list(result[_].keys())
        values = list(result[_].values())
        plt.subplot(flag)
        plt.title(_)
        plt.ylim(0, 1.1)
        plt.xticks(range(len(keys)), keys)
        plt.scatter(range(len(values)), values)
        for a, b in zip(range(len(keys)), values):
            print(a, b)
            plt.text(a, b + 0.01, "%.4f" % b, ha='center', va='bottom', fontsize=7)
        flag += 1
    plt.show()


def main(is_draw):
    dataset = ['PB', 'USAir']
    result = {}
    for _ in dataset:
        training_set = np.loadtxt('dataset/' + _ + '_train.txt')
        test_set = np.loadtxt('dataset/' + _ + '_test.txt')
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
        result[_] = auc_result
        print(_, ':', auc_result)
    print(result)
    if is_draw:
        draw(dataset, result)


if __name__ == '__main__':
    # main(False)
    draw()
