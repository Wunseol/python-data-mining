"""C4.5决策树算法模块

基于信息增益率（Gain Ratio）替代信息增益作为分裂准则，
支持离散特征与连续特征的二分处理，克服ID3偏向多值特征的问题。
"""
from math import log, sqrt
import operator
import re

"""
C4.5 决策树算法

C4.5 是 ID3 的改进版本，使用信息增益率（Gain Ratio）替代信息增益作为分裂准则，
克服了 ID3 偏向选择取值较多特征的问题。信息增益率 = 信息增益 / 固有信息值（Split Information）。
C4.5 还支持连续特征的二分处理。

本模块功能：
- calcShannonEntropy: 计算数据集的香农熵
- chooseBestFeat: 选择信息增益率最大的特征（支持离散和连续特征）
- splitDataSet / splitContinuousDataSet: 离散/连续特征的数据集划分
- createTree: 递归构建 C4.5 决策树
"""


def createDataSet():
    dataSet = [[1, '长', '粗', '男'],
               [2, '短', '粗', '男'],
               [3, '短', '粗', '男'],
               [4, '长', '细', '女'],
               [5, '短', '细', '女'],
               [6, '短', '粗', '女'],
               [7, '长', '粗', '女'],
               [8, '长', '粗', '女']]
    labels = ['序号', '头发', '声音']
    return dataSet, labels


def classCount(dataSet):
    labelCount = {}
    for one in dataSet:
        if one[-1] not in labelCount.keys():
            labelCount[one[-1]] = 0
        labelCount[one[-1]] += 1
    return labelCount


def calcShannonEntropy(dataSet):
    labelCount = classCount(dataSet)
    numEntries = len(dataSet)
    Entropy = 0.0
    for i in labelCount:
        prob = float(labelCount[i]) / numEntries
        Entropy -= prob * log(prob, 2)
    return Entropy


def majorityClass(dataSet):
    labelCount = classCount(dataSet)
    sortedLabelCount = sorted(labelCount.items(), key=operator.itemgetter(1),reverse=True)
    return sortedLabelCount[0][0]


def splitDataSet(dataSet, i, value):
    subDataSet = []
    for one in dataSet:
        if one[i] == value:
            reduceData = one[:i]
            reduceData.extend(one[i + 1:])
            subDataSet.append(reduceData)
    return subDataSet


def splitContinuousDataSet(dataSet, i, value, direction):
    subDataSet = []
    for one in dataSet:
        if direction == 0:
            if one[i] > value:
                reduceData = one[:1]
                reduceData.extend(one[i + 1:])
                subDataSet.append(reduceData)
        if direction == 1:
            if one[i] <= value:
                reduceData = one[:i]
                reduceData.extend(one[i + 1:])
                subDataSet.append(reduceData)
    return subDataSet


def chooseBestFeat(dataSet, labels):
    baseEntropy = calcShannonEntropy(dataSet)
    bestFeat = 0
    baseGainRatio = -1
    numFeats = len(dataSet[0]) - 1
    bestSplitDic = {}
    i = 0
    print('dataSet[0]:' + str(dataSet[0]))
    for i in range(numFeats):
        featVals = [example[i] for example in dataSet]
        if type(featVals[0]).__name__ == 'float' or type(featVals[0]).__name__ == 'int':
            j = 0
            sortedFeatVals = sorted(featVals)
            splitList = []
            for j in range(len(featVals) - 1):
                splitList.append((sortedFeatVals[j] + sortedFeatVals[j + 1]) / 2.0)
            for j in range(len(splitList)):
                newEntropy = 0.0
                gainRatio = 0.0
                splitInfo = 0.0
                value = splitList[j]
                subDataSet0 = splitContinuousDataSet(dataSet, i, value, 0)
                subDataSet1 = splitContinuousDataSet(dataSet, i, value, 1)
                prob0 = float(len(subDataSet0)) / len(dataSet)
                newEntropy -= prob0 * calcShannonEntropy(subDataSet0)
                prob1 = float(len(subDataSet1)) / len(dataSet)
                newEntropy -= prob1 * calcShannonEntropy(subDataSet1)
                splitInfo -= prob0 * log(prob0, 2)
                splitInfo -= prob1 * log(prob1, 2)
                gainRatio = float(baseEntropy - newEntropy) / splitInfo
                print('IVa' + str(j) + ':' + str(splitInfo))
                if gainRatio > baseGainRatio:
                    baseGainRatio = gainRatio
                    bestSplit = j
                    bestFeat = i
            bestSplitDic[labels[i]] = splitList[bestSplit]
        else:
            uniqueFeatVals = set(featVals)
            GainRatio = 0.0
            splitInfo = 0.0
            newEntropy = 0.0
            for value in uniqueFeatVals:
                subDataSet = splitDataSet(dataSet, i, value)
                prob = float(len(subDataSet)) / len(dataSet)
                splitInfo -= prob * log(prob, 2)
                newEntropy -= prob * calcShannonEntropy(subDataSet)
            gainRatio = float(baseEntropy - newEntropy) / splitInfo
            if gainRatio > baseGainRatio:
                bestFeat = i
                baseGainRatio = gainRatio
    if type(dataSet[0][bestFeat]).__name__ == 'float' or type(dataSet[0][bestFeat]).__name__ == 'int':
        bestFeatValue = bestSplitDic[labels[bestFeat]]
    if type(dataSet[0][bestFeat]).__name__ == 'str':
        bestFeatValue = labels[bestFeat]
    return bestFeat, bestFeatValue


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if len(set(classList)) == 1:
        return classList[0][0]
    if len(dataSet[0]) == 1:
        return majorityClass(dataSet)
    Entropy = calcShannonEntropy(dataSet)
    bestFeat, bestFeatLabel = chooseBestFeat(dataSet, labels)
    print('bestFeat:' + str(bestFeat) + '--' + str(labels[bestFeat]) + ',bestFeatLabel:' + str(bestFeatLabel))
    myTree = {labels[bestFeat]: {}}
    subLabels = labels[:bestFeat]
    subLabels.extend(labels[bestFeat + 1:])
    print('subLabels:' + str(subLabels))
    if type(dataSet[0][bestFeat]).__name__ == 'str':
        featVals = [example[bestFeat] for example in dataSet]
        uniqueVals = set(featVals)
        print('uniqueVals:' + str(uniqueVals))
        for value in uniqueVals:
            reduceDataSet = splitDataSet(dataSet, bestFeat, value)
            print('reduceDataSet:' + str(reduceDataSet))
            myTree[labels[bestFeat]][value] = createTree(reduceDataSet, subLabels)
    if type(dataSet[0][bestFeat]).__name__ == 'int' or type(dataSet[0][bestFeat]).__name__ == 'float':
        value = bestFeatLabel
        greaterDataSet = splitContinuousDataSet(dataSet, bestFeat, value, 0)
        smallerDataSet = splitContinuousDataSet(dataSet, bestFeat, value, 1)
        print('greaterDataset:' + str(greaterDataSet))
        print('smallerDataSet:' + str(smallerDataSet))
        print('==' * len(dataSet[0]))
        myTree[labels[bestFeat]]['>' + str(value)] = createTree(greaterDataSet, subLabels)
        print(myTree)
        print('=='*len(dataSet[0]))
        myTree[labels[bestFeat]]['<=' + str(value)] = createTree(smallerDataSet, subLabels)
    return myTree


if __name__ == '__main__':
    print('20211113492  陈文聪')
    dataSet, labels = createDataSet()
    print(createTree(dataSet, labels))


