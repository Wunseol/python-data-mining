# -*- coding: utf-8 -*-
"""
FP-Growth算法（频繁模式增长算法）

FP-Growth算法通过构建FP树（频繁模式树）来发现频繁项集，
相比Apriori算法只需扫描数据库两次，效率更高。

主要包含：
1. FP树的类定义（treeNode）
2. FP树构建（createTree）
3. 条件模式基发现（findPrefixPath）
4. 频繁项集递归挖掘（mineTree）

@author: wzy
"""


class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}

    def inc(self, numOccur):
        self.count += numOccur

    def disp(self, ind=1):
        print('  '*ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind + 1)


# ==============================================================================
# FP树构建
# ==============================================================================

def createTree(dataSet, minSup=1):
    headerTable = {}
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    lessThanMinsup = list(filter(lambda k: headerTable[k] < minSup, headerTable.keys()))
    for k in lessThanMinsup:
        del(headerTable[k])
    for k in list(headerTable):
        if headerTable[k] < minSup:
            del(headerTable[k])
    freqItemSet = set(headerTable.keys())
    if len(freqItemSet) == 0:
        return None, None
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]
    retTree = treeNode('Null Set', 1, None)
    for tranSet, count in dataSet.items():
        localD = {}
        for item in tranSet:
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            ordereItems = [v[0] for v in sorted(localD.items(), key=lambda p: (p[1], p[0]), reverse=True)]
            updateTree(ordereItems, retTree, headerTable, count)
    return retTree, headerTable


def updateTree(items, inTree, headerTable, count):
    if items[0] in inTree.children:
        inTree.children[items[0]].inc(count)
    else:
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        if headerTable[items[0]][1] is None:
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:
        updateTree(items[1:], inTree.children[items[0]], headerTable, count)


def updateHeader(nodeToTest, targetNode):
    while(nodeToTest.nodeLink is not None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode


# ==============================================================================
# 数据加载与初始化
# ==============================================================================

def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat


def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        fset = frozenset(trans)
        retDict.setdefault(fset, 0)
        retDict[fset] += 1
    return retDict


# ==============================================================================
# 条件模式基与频繁项集挖掘
# ==============================================================================

def ascendTree(leafNode, prefixPath):
    if leafNode.parent is not None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)


def findPrefixPath(basePat, treeNode):
    condPats = {}
    while treeNode is not None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats


def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: str(p[1]))]
    for basePat in bigL:
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        myContTree, myHead = createTree(condPattBases, minSup)
        if myHead is not None:
            print('conditional tree for: ', newFreqSet)
            myContTree.disp(1)
            mineTree(myContTree, myHead, minSup, newFreqSet, freqItemList)


if __name__ == '__main__':
    simpDat = loadSimpDat()
    initSet = createInitSet(simpDat)
    myFPtree, myHeaderTab = createTree(initSet, 3)
    myFPtree.disp()
    myFreqList = []
    mineTree(myFPtree, myHeaderTab, 3, set([]), myFreqList)
    print("频繁项集：", myFreqList)
