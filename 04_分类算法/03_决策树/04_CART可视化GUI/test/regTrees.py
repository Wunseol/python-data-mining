import numpy as np
import matplotlib.pyplot as plt


def loadDataSet(fileName):
    dataMat = []
    with open(fileName) as fr:
        for line in fr.readlines():
            curLine = line.strip().split('\t')
            fltLine = list(map(float, curLine))

            dataMat.append(fltLine)
    return dataMat


def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]

    return mat0, mat1


def regLeaf(dataSet):
    return np.mean(dataSet[:, -1])


def regErr(dataSet):
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]


def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    tolS = ops[0]
    tolN = ops[1]

    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)

    m, n = np.shape(dataSet)
    S = errType(dataSet)

    bestS = np.inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n - 1):

        for splitVal in set((dataSet[:, featIndex].T.A.tolist())[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN): continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex, bestValue


def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat is None: return val
    retTree = {'spInd': feat, 'spVal': val}

    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


def showDataSet():
    myDat = loadDataSet('ex00.txt')

    myMat = np.mat(myDat)
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(6, 19))
    axs[0].scatter(myMat[:, 0].flatten().A[0], myMat[:, 1].flatten().A[0], c='blue')
    myDat1 = loadDataSet('ex0.txt')

    myMat1 = np.mat(myDat1)
    axs[1].scatter(myMat1[:, 1].flatten().A[0], myMat1[:, 2].flatten().A[0], c='blue')
    myDat2 = loadDataSet('ex2.txt')

    myMat2 = np.mat(myDat2)
    axs[2].scatter(myMat2[:, 0].flatten().A[0], myMat2[:, 1].flatten().A[0], c='blue')
    plt.show()


def isTree(obj):
    return type(obj).__name__ == 'dict'


def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0


def prune(tree, testData):
    if np.shape(testData)[0] == 0: return getMean(tree)
    if isTree(tree['right']) or isTree(tree['left']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']): tree['right'] = prune(tree['right'], rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = np.sum(np.power(lSet[:, -1] - tree['left'], 2)) + \
                       np.sum(np.power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = np.sum(np.power(testData[:, -1] - treeMean, 2))

        if errorMerge < errorNoMerge:
            print("merging")
            return treeMean
        else:
            return tree
    else:
        return tree


# 模型树
def linearSolve(dataSet):
    m, n = np.shape(dataSet)
    X = np.mat(np.ones((m, n)))
    Y = np.mat(np.ones((m, 1)))
    X[:, 1:n] = dataSet[:, 0:n - 1]
    Y = dataSet[:, -1]
    xTx = X.T * X
    if np.linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y


def modelLeaf(dataSet):
    ws, X, Y = linearSolve(dataSet)
    return ws


def modelErr(dataSet):
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return np.sum(np.power(Y - yHat, 2))


# 用树回归进行预测
def regTreeEval(model, inDat):
    return float(model)


def modelTreeEval(model, inDat):
    n = np.shape(inDat)[1]
    X = np.mat(np.ones((1, n + 1)))
    X[:, 1:n + 1] = inDat

    return float((X * model).item())


def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree): return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)


# 对数据进行树结构建模
def createForeCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = np.mat(np.zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, np.mat(testData[i]), modelEval)
    return yHat


if __name__ == '__main__':
    trainMat = np.mat(loadDataSet('bikeSpeedVsIq_train.txt'))
    testMat = np.mat(loadDataSet('bikeSpeedVsIq_test.txt'))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(testMat[:, 0].A.reshape(1, -1), testMat[:, 1].A.reshape(1, -1), c='blue')
    plt.xlabel('speed of riding')
    plt.ylabel('IQ')
    plt.show()

    myTree = createTree(trainMat, ops=(1, 20))

    yHat = createForeCast(myTree, testMat[:, 0])
    print('回归树的相关性R^2：', np.corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1])

    myTree1 = createTree(trainMat, modelLeaf, modelErr, ops=(1, 20))
    yHat = createForeCast(myTree1, testMat[:, 0], modelTreeEval)
    print('模型树的相关性R^2：', np.corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1])

    ws, X, Y = linearSolve(trainMat)
    print('ws=', ws)

    for i in range(np.shape(testMat)[0]):
        yHat[i] = testMat[i, 0] * ws[1, 0] + ws[0, 0]
    print('标准线性回归的相关性R^2：', np.corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1])
#   标准线性回归的相关性R^2： 0.9434684235674762
print("20211113496刘育豪")