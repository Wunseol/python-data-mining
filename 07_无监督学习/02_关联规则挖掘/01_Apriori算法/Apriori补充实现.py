#-*-coding:utf-8-*-
def loadDataSet():      #创建数据集
    return [[1,3, 4],[2, 3, 5],[1, 2, 3, 5, 6],[2, 5], [1, 3, 5]]
def createC1(dataSet):      #创建数据集中所有单一元素组成的集合
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset, C1))
    # frozenset 返回一个冻结的集合，不能添加或删除任何元素
def scanD(D, Ck, minSupport):
    ssCnt = {}
    # 统计每一个元素出现的次数
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if not can in ssCnt:
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(len(D))        #获取样本中的元素个数
    retList = []
    supportData = {}
    for key in ssCnt:       #遍历每一个元素
        support = ssCnt[key] / numItems     #计算每一个元素的支持度
        if support >= minSupport:       #若支持度大于或等于最小支持唐
            retList.insert(0, key)      #将指定对象插人列表的指定位置
        supportData[key] = support
    return retList, supportData
def aprioriGen(Lk, k):      #组合向上合并
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):       #两两组合遍历
            L1 = list(Lk[i])[:k - 2]
            L2 = list(Lk[j])[:k - 2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList
def apriori(dataSet,minSupport=0.5):        #Apriori算法
    C1 = createC1(dataSet)      #将数据集中所有单一元素组成的集合保存到 Cl中
    D = list(map(set, dataSet))     #将数据集元素为 set 集合，然后将结果保存为列表
    L1, supportData = scanD(D, C1, 0.5)
    # 从C1生成 L1并返回由符合条件的元素及其支持度组成的字典
    L = [L1]        #将符合条件的元素转换为列表保存在工中
    k = 2
    # I[n]就代表 n+1个元素的集合,例如 L[O]代表 1个元素的集合
    while (len(L[k - 2]) > 0):
        Ck = aprioriGen(L[k - 2], k)
        Lk, supK = scanD(D, Ck, minSupport)
        # dict.update(dict2)        #update函数把字典 dict2的键-值对更新到 dict中
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData
def generateRules(L, supportData, minConf=0.7):      #生成关联规则
    bigRuleList = []        #存储所有的关联规则
    # 只获取两个或更多集合的项目
    # 集合有两个或两个以上才可能存在关联
    for i in range(1, len(L)):
        for freqSet in L[i]:
            # 遍历L中的每一个频繁项集并为其创建只包含单个元素集合的列表 H1
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList
# 生成候选规则集合,计算规则的置信度,并找到满足最小置信度要求的规则
def calcConf(freqSet, H, supportData, br1, minConf=0.7):
    prunedH = []
    for conseq in H:        #遍历L中的某个i-频繁项集的每个元素
        # 置信度计算，结合支持度数据
        conf = supportData[freqSet] / supportData[freqSet - conseq]
        if conf >= minConf:
            # 如果某条规则满足最小置信度的要求,那么将这条规则输出到屏幕
            print(freqSet - conseq, '-->', conseq, 'conf:', conf)
            # 添加到规则中,br1是前面通过检查的 bigRuleList
            br1.append((freqSet - conseq, conseq, conf))
            prunedH.append(conseq)      #保存通过检查的项
    return prunedH
# 生成候选规则集合,计算规则的置信度,并找到满足最小置信度要求的规则
def rulesFromConseq(freqSet, H, supportData, br1, minConf=0.7):
    m = len(H[0])
    if (len(freqSet) > (m + 1)):        #频繁项集元素个数大于单个集合的元素个数
        # 存在顺序不同、元素相同的集合,合并具有相同元素的集合
        Hmp1 = aprioriGen(H, m + 1)
        Hmp1 = calcConf(freqSet, Hmp1, supportData, br1, minConf)       #计算置信度
        # 满足最小置信度要求的规则多于一条,则递归判断是否可以进一步组合这些规则
        if (len(Hmp1) > 1):
            rulesFromConseq(freqSet, Hmp1, supportData, br1, minConf)

dataSet = loadDataSet()
L, supportData = apriori(dataSet, 0.5)
rules = generateRules(L, supportData, minConf=0.7)
print(L)

