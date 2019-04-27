import numpy as np


def Calculatain_AUC(Train_MatrixAdjacency, Test_MatrixAdjacency,
                    SimilarityMatrix,MaxNode):
    AUCnum = 60000
    SimilarityMatrix = np.triu(SimilarityMatrix -
                               SimilarityMatrix * Train_MatrixAdjacency)

    Matrix_NoExist = np.ones(MaxNode) - Train_MatrixAdjacency - Test_MatrixAdjacency - np.eye(MaxNode)

    Test = np.triu(Test_MatrixAdjacency)
    NoExist = np.triu(Matrix_NoExist)

    Test_num = len(np.argwhere(Test == 1))
    NoExist_num = len(np.argwhere(NoExist == 1))

    Test_random = [
        int(x)
        for index, x in enumerate((Test_num * np.random.rand(1, AUCnum))[0])
    ]
    NoExist_random = [
        int(x)
        for index, x in enumerate((NoExist_num * np.random.rand(1, AUCnum))[0])
    ]

    TestPrediction = SimilarityMatrix * Test
    NoExistPrediction = SimilarityMatrix * NoExist

    TestIndex = np.argwhere(Test == 1)
    TestData = np.array(
        [TestPrediction[x[0], x[1]] for index, x in enumerate(TestIndex)]).T

    NoExistIndex = np.argwhere(NoExist == 1)
    NoExistData = np.array([
        NoExistPrediction[x[0], x[1]] for index, x in enumerate(NoExistIndex)
    ]).T

    Test_random = np.array(
        [TestData[x] for index, x in enumerate(Test_random)])
    NoExist_random = np.array(
        [NoExistData[x] for index, x in enumerate(NoExist_random)])

    n1, n2 = 0, 0

    for num in range(AUCnum):
        if Test_random[num] > NoExist_random[num]:
            n1 += 1
        elif Test_random[num] == NoExist_random[num]:
            n2 += 0.5
        else:
            n1 += 0
    AUC = (n1 + n2) / AUCnum

    return AUC
