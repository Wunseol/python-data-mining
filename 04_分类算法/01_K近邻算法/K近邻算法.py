"""K近邻算法（K-Nearest Neighbors, KNN）模块

实现KNN分类算法，包括距离计算、多数投票分类、数据归一化等功能，
并提供约会数据分类与手写数字识别的完整示例。
"""
import os
'''
Created on Sep 16, 2010
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)

Output:     the most popular class label

@author: pbharrin
'''
import sys
import numpy as np
import operator
from os import listdir

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from utils import setup_chinese_font

import matplotlib.pyplot as plt
setup_chinese_font()

"""
K近邻算法（K-Nearest Neighbors, KNN）

KNN 是一种基于实例的惰性学习算法，其核心思想是：对于待分类样本，
在训练集中找到距离最近的 k 个邻居，通过多数投票决定其类别。

本模块功能：
- classify0: 核心分类函数，基于欧氏距离与多数投票实现 KNN 分类
- createDataSet: 创建简单示例数据集
- file2matrix: 从文件读取约会数据
- autoNorm: 数值归一化
- datingClassTest: 约会数据分类测试
- img2vector / handwritingClassTest: 手写数字识别
"""

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def file2matrix(filename):
    with open(filename) as fr:
        numberOfLines = len(fr.readlines())
    returnMat = np.zeros((numberOfLines,3))
    classLabelVector = []
    with open(filename) as fr:
        index = 0
        for line in fr.readlines():
            line = line.strip()
            listFromLine = line.split('\t')
            returnMat[index,:] = listFromLine[0:3]
            classLabelVector.append(int(listFromLine[-1]))
            index += 1
    return returnMat,classLabelVector

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m,1))
    normDataSet = normDataSet/np.tile(ranges, (m,1))   #element wise divide
    return normDataSet, ranges, minVals

def datingClassTest():
    hoRatio = 0.50      #hold out 10%
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print(("the classifier came back with: %d, the real answer is: %d") % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print(("the total error rate is: %f") % (errorCount/float(numTestVecs)))
    print(errorCount)


# 基础1_k近邻算法识别手写数字
def img2vector(filename):
    returnVect = np.zeros((1,1024))
    with open(filename) as fr:
        for i in range(32):
            lineStr = fr.readline()
            for j in range(32):
                returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir(os.path.join(_SCRIPT_DIR, 'digits', 'trainingDigits'))           #load the training set
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector(os.path.join(_SCRIPT_DIR, 'digits', 'trainingDigits', fileNameStr))
    testFileList = listdir(os.path.join(_SCRIPT_DIR, 'digits', 'testDigits'))        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector(os.path.join(_SCRIPT_DIR, 'digits', 'testDigits', fileNameStr))
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print(("the classifier came back with: %d, the real answer is: %d") % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print(("\nthe total number of errors is: %d") % errorCount)
    print(("\nthe total error rate is: %f") % (errorCount/float(mTest)))


def plot_decision_boundary():
    from sklearn.datasets import make_classification
    from sklearn.neighbors import KNeighborsClassifier

    np.random.seed(42)
    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                               n_informative=2, n_clusters_per_class=1, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X, y)

    h = 0.05
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='k', s=40)
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.title('KNN决策边界 (K=5)')
    plt.colorbar(scatter, label='类别')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    print("==== K近邻算法演示 ====")

    print("\n--- 使用自实现 classify0 进行分类 ---")
    group, labels = createDataSet()
    test_vec = [0.2, 0.3]
    result = classify0(test_vec, group, labels, 3)
    print("测试向量 %s 的分类结果: %s" % (test_vec, result))

    print("\n--- 使用 sklearn KNeighborsClassifier 对比 ---")
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(group, labels)
    result_sk = knn.predict([test_vec])
    print("sklearn KNeighborsClassifier 分类结果: %s" % result_sk[0])

    plot_decision_boundary()
