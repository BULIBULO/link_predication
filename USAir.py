import numpy as np
import AUC
from Algorithm import *
import MatrixAdjacency
import matplotlib.pyplot as plt

Training_set = np.loadtxt('USAir_train.txt')
Test_set = np.loadtxt('USAir_test.txt')
MaxNode = int(Training_set.max() + 1)
Train_Matrix_Adjacency = MatrixAdjacency.TransfromMatrix(Training_set, MaxNode)
Test_Matrix_Adjacency = MatrixAdjacency.TransfromMatrix(Test_set, MaxNode)

Algorithms = [CN, Jaccard, RA, AA, PA, Katz]
SimilarityMatrix = []
AUCresult = {"CN":0,"Jaccard":0,"RA":0,"AA":0,"PA":0,"Katz":0}
# a = Algorithm.CN(Train_Matrix_Adjacency)
# b = AUC.Calculatain_AUC(Train_Matrix_Adjacency,Test_Matrix_Adjacency,a,MaxNode)
# print(b)
for i in Algorithms:
    SimilarityMatrix.append(i(Train_Matrix_Adjacency))
    temp = i(Train_Matrix_Adjacency)
    AUCresult[i.__name__] = AUC.Calculatain_AUC(Train_Matrix_Adjacency,Test_Matrix_Adjacency,temp,MaxNode)

print(AUCresult)
