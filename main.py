from auc import calculating_auc
from algorithm import *
from matrix_adjacency import transform_matrix
from matplotlib import pyplot as plt


def draw(datasets=None, result=None):
    
    if datasets is None: 
        # test dataset of drawing 
        datasets = ['PB', 'USAir']
        result = {
            'PB': {'cn': 0.9311833333333334, 'jc': 0.743475, 'ra': 0.843675, 'aa': 0.88715, 'pa': 0.5, 'katz': 0.9385},
            'USAir': {'cn': 0.9623, 'jc': 0.8017333333333333, 'ra': 0.8901666666666667, 'aa': 0.9239, 'pa': 0.5,
                    'katz': 0.9611}}
    
    fig, axes = plt.subplots(1, len(datasets), sharey=True,
                             figsize=(6*len(datasets), 6))
    axes[0].set_ylabel("AUC")
    for i, dataset in enumerate(datasets):
        keys = list(result[dataset].keys())
        values = list(result[dataset].values())
        axes[i].set_ylim(0, 1.1)
        axes[i].set_title(dataset)
        axes[i].set_xticklabels([""]+keys)
        axes[i].scatter(range(len(values)), values)
        for a, b in zip(range(len(keys)), values):
            print(a, b)
            axes[i].text(a, b + 0.01, "%.4f" %
                         b, ha='center', va='bottom', fontsize=7)
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
            auc_result[i.__name__] = calculating_auc(
                train_matrix_adjacency, test_matrix_adjacency, temp, max_node)
        result[_] = auc_result
        print(_, ':', auc_result)
    print(result)
    if is_draw:
        draw(dataset, result)


if __name__ == '__main__':
    main(True)
    # draw()
