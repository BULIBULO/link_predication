from auc import calculating_auc
from algorithm import *
from matrix_adjacency import transform_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold


def draw(datasets=None, result=None):
    if datasets is None:
        datasets = ['PB', 'USAir']
        result = {
            'PB': {'cn': 0.9311833333333334, 'jc': 0.743475, 'ra': 0.843675, 'aa': 0.88715, 'pa': 0.5, 'katz': 0.9385},
            'USAir': {'cn': 0.9623, 'jc': 0.8017333333333333, 'ra': 0.8901666666666667, 'aa': 0.9239, 'pa': 0.5,
                      'katz': 0.9611}}

    fig, axes = plt.subplots(1, len(datasets), sharey=True, figsize=(6 * len(datasets), 6))
    axes[0].set_ylabel("AUC")
    for i, dataset in enumerate(datasets):
        keys = list(result[dataset].keys())
        values = list(result[dataset].values())
        axes[i].set_ylim(0, 1.1)
        axes[i].set_title(dataset)
        axes[i].set_xticklabels([""] + keys)
        axes[i].scatter(range(len(values)), values)
        for a, b in zip(range(len(keys)), values):
            print(a, b)
            axes[i].text(a, b + 0.01, "%.4f" % b, ha='center', va='bottom', fontsize=7)
    plt.show()


def main(is_draw):
    datasets = ['USAir', 'PB', 'NS', 'Grid', 'INT', 'PPI']
    result = {}
    for _ in datasets:
        data = np.loadtxt('dataset/' + _ + '.txt')
        data = data[:, :2]
        kf = KFold(n_splits=5)
        kf_index = 0
        kf_result = {}
        max_node = int(data.max() + 1)
        for train_index, test_index in kf.split(data):
            train_set = data[train_index]
            test_set = data[test_index]

            train_matrix_adjacency = transform_matrix(train_set, max_node)
            test_matrix_adjacency = transform_matrix(test_set, max_node)

            algorithms = [cn, jc, ra, aa, pa, katz]
            similarity_matrix = []
            auc_result = {}
            for i in algorithms:
                similarity_matrix.append(i(train_matrix_adjacency))
                temp = i(train_matrix_adjacency)
                auc_result[i.__name__] = calculating_auc(train_matrix_adjacency, test_matrix_adjacency, temp, max_node)
            kf_result[kf_index] = auc_result
            kf_index += 1
        print(_, ':', kf_result)
        result[_] = kf_result
    print(result)
    if is_draw:
        draw(datasets, result)


if __name__ == '__main__':
    main(False)
    # draw()
