"""朴素贝叶斯算法（Naive Bayes）模块

基于贝叶斯定理与特征独立性假设，实现文本分类与垃圾邮件检测，
包括词表构建、词袋模型、拉普拉斯平滑训练与对数概率分类。
"""
import os
'''
Created on Oct 19, 2010

@author: Peter
'''
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from utils import setup_chinese_font

import matplotlib.pyplot as plt
setup_chinese_font()

# 获取当前脚本所在目录，用于定位数据文件
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

"""
朴素贝叶斯算法（Naive Bayes）

朴素贝叶斯基于贝叶斯定理，假设特征之间相互独立（"朴素"假设），
通过计算后验概率 P(类别|特征) 进行分类。

本模块功能：
- trainNB0: 训练朴素贝叶斯分类器（拉普拉斯平滑 + 对数概率）
- classifyNB: 对新样本进行分类
- createVocabList / setOfWords2Vec / bagOfWords2VecMN: 词表构建与向量化
- spamTestSklearn: 使用内嵌合成数据进行垃圾邮件分类演示（无外部依赖）
- spamTest: 基于外部邮件文件的垃圾邮件分类测试
"""

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet = set([])  #create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) #union of the two sets
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print (("the word: %s is not in my Vocabulary!") % word)
    return returnVec

def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = np.sum(trainCategory)/float(numTrainDocs)
    p0Num = np.ones(numWords); p1Num = np.ones(numWords)      #change to ones()
    p0Denom = 2.0; p1Denom = 2.0                        #change to 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += np.sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += np.sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom)          #change to log()
    p0Vect = np.log(p0Num/p0Denom)          #change to log()
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = np.sum(vec2Classify * p1Vec) + np.log(pClass1)    #element-wise mult
    p0 = np.sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(np.array(trainMat),np.array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print (testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print (testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))

def textParse(bigString):    #input is big string, #output is word list
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    # 注意：此函数依赖外部 email/spam/ 和 email/ham/ 目录下的数据文件，
    # 推荐使用 spamTestSklearn() 替代，该函数使用内嵌合成数据，无需外部文件。
    docList=[]; classList = []; fullText =[]
    for i in range(1,26):
        wordList = textParse(open(os.path.join(_SCRIPT_DIR, 'email', 'spam', '%d.txt' % i), errors='ignore').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open(os.path.join(_SCRIPT_DIR, 'email', 'ham', '%d.txt' % i), errors='ignore').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)#create vocabulary
    trainingSet = list(range(50)); testSet=[]           #create test set
    for i in range(10):
        randIndex = int(np.random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(np.array(trainMat),np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print ("classification error",docList[docIndex])
    print ('the error rate is: ',float(errorCount)/len(testSet))
    #return vocabList,fullText

# ==== 使用内嵌合成数据进行垃圾邮件分类测试（无外部文件依赖） ====

def spamTestSklearn():
    """
    使用内嵌的合成中英文邮件数据演示朴素贝叶斯垃圾邮件分类。
    无需外部数据文件，满足项目自包含要求（NFR2）。
    复用现有函数：createVocabList, bagOfWords2VecMN, trainNB0, classifyNB
    随机种子固定为 random_state=42 以保证结果可复现。
    注意：由于原 textParse 使用 \\W* 正则无法正确处理中文，
    此函数内部定义了 _text_parse_cn_en 来同时支持中英文分词。
    """

    # ==== 中英文混合文本解析器 ====
    # 原 textParse 使用 re.split(r'\W*', ...) 无法正确处理中文字符，
    # 因为 \W 不匹配 CJK 字符，导致中文被拆成单字后又被 len>2 过滤掉。
    # 此解析器使用 findall 分别提取英文单词和连续中文字符串作为词元。
    import re as _re
    def _text_parse_cn_en(big_string):
        tokens = _re.findall(r'[a-zA-Z]{3,}', big_string)
        tokens += _re.findall(r'[\u4e00-\u9fff]{2,}', big_string)
        return [t.lower() for t in tokens]

    # ==== 内嵌合成垃圾邮件数据（spam, class=1） ====
    spam_docs = [
        "Free money! Click here to claim your prize now. You have won a lottery ticket worth one million dollars. Act now before it expires!",
        "Buy cheap medications online without prescription. Best prices on viagra, cialis and more. Order today and get free shipping worldwide!",
        "Congratulations! You have been selected as our lucky winner. Click the link below to receive your free gift card worth five hundred dollars.",
        "Make money fast! Work from home and earn thousands every week. No experience needed. Sign up today and start earning immediately!",
        "Discount offer: 90 percent off on all luxury watches. Replica rolex, omega and more. Limited time offer, buy now and save big!",
        "免费领取大奖！点击链接立即领取您的百万现金奖励，名额有限先到先得，不要错过这个千载难逢的机会！",
        "恭喜您中奖了！您已被随机抽选为本期幸运用户，点击领取价值五千元购物卡，活动即将截止请尽快操作！",
        "特价促销：各类名牌手表一折起，劳力士欧米茄应有尽有，限时抢购包邮到家，立即下单享受超低折扣！",
        "在家轻松赚钱！无需经验每天收入过千，加入我们的团队开始赚钱之旅，名额有限立即注册！",
        "免费试用减肥药！一周瘦十斤不是梦，纯天然成分无副作用，现在订购还送瘦身手册，点击了解详情！",
        "Hot stock tip! Buy penny shares now and watch your investment triple in just one week. Insider information guaranteed profits act fast!",
        "Your account has been compromised. Click here immediately to verify your identity and secure your account before it is too late!",
    ]

    # ==== 内嵌合成正常邮件数据（ham, class=0） ====
    ham_docs = [
        "Hi team, just a reminder that our weekly project meeting is scheduled for tomorrow at ten in conference room three. Please prepare your updates.",
        "Dear colleague, I have attached the quarterly report for your review. Please let me know if you have any questions or suggestions before Friday.",
        "Hey, are you free for lunch this Saturday? I was thinking we could try that new Italian restaurant downtown. Let me know what time works for you.",
        "Good morning, I wanted to follow up on our discussion from yesterday regarding the database migration plan. I have outlined the next steps below.",
        "Hello professor, I have completed the assignment and submitted it through the online portal. Could you please confirm that you received it? Thank you.",
        "各位同事好，本周五下午两点在三楼会议室召开项目进度汇报会，请各位提前准备好各自负责模块的进展报告，谢谢配合。",
        "你好，附件是本季度的财务报表，请在周三之前审阅并反馈意见，如有疑问请随时联系我，辛苦了！",
        "周末有空一起吃饭吗？听说公司附近新开了一家日料店评价不错，要不要一起去试试？时间你来定就行。",
        "您好，关于上次讨论的系统升级方案，我已经整理了详细的技术文档，请查收附件并安排时间讨论后续计划。",
        "提醒：明天上午十点有客户来访，请相关部门提前做好准备，会议室已预定完毕，如有变动请及时通知。",
        "Hi mom, I just wanted to let you know that I arrived safely. The flight was smooth and the weather here is beautiful. I will call you this weekend.",
        "Dear hiring manager, thank you for the opportunity to interview yesterday. I enjoyed learning about the position and look forward to hearing from you.",
    ]

    # ==== 解析文本并构建文档列表与类别标签 ====
    docList = []
    classList = []
    fullText = []

    for doc in spam_docs:
        word_list = _text_parse_cn_en(doc)
        docList.append(word_list)
        fullText.extend(word_list)
        classList.append(1)  # spam 标记为 1

    for doc in ham_docs:
        word_list = _text_parse_cn_en(doc)
        docList.append(word_list)
        fullText.extend(word_list)
        classList.append(0)  # ham 标记为 0

    # ==== 构建词汇表 ====
    vocabList = createVocabList(docList)

    # ==== 划分训练集与测试集（固定随机种子 random_state=42） ====
    import random as _random
    _random.seed(42)
    total_docs = len(docList)
    trainingSet = list(range(total_docs))
    testSet = []
    # 随机选取约 30% 的样本作为测试集
    test_count = int(total_docs * 0.3)
    if test_count < 1:
        test_count = 1
    for _ in range(test_count):
        randIndex = int(_random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])

    # ==== 训练朴素贝叶斯分类器 ====
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))

    # ==== 在测试集上评估分类器 ====
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print("classification error", docList[docIndex])
    error_rate = float(errorCount) / len(testSet)
    print('the error rate is: ', error_rate)
    return error_rate

def calcMostFreq(vocabList,fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]

def localWords(feed1,feed0):
    import feedparser
    docList=[]; classList = []; fullText =[]
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1) #NY is class 1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)#create vocabulary
    top30Words = calcMostFreq(vocabList,fullText)   #remove top 30 words
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    trainingSet = range(2*minLen); testSet=[]           #create test set
    for i in range(20):
        randIndex = int(np.random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(np.array(trainMat),np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print ('the error rate is: ',float(errorCount)/len(testSet))
    return vocabList,p0V,p1V

def getTopWords(ny,sf):
    import operator
    vocabList,p0V,p1V=localWords(ny,sf)
    topNY=[]; topSF=[]
    for i in range(len(p0V)):
        if p0V[i] > -6.0 : topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -6.0 : topNY.append((vocabList[i],p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print ("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print (item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print ("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print (item[0])


if __name__ == '__main__':
    print("==== 朴素贝叶斯算法演示 ====")

    print("\n--- 使用内嵌合成数据进行垃圾邮件分类 ---")
    error_rate = spamTestSklearn()
    print("垃圾邮件分类错误率: %.4f" % error_rate)

    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB as SklearnGaussianNB

    np.random.seed(42)
    X, y = make_classification(n_samples=300, n_features=2, n_redundant=0,
                               n_informative=2, n_clusters_per_class=1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    sk_nb = SklearnGaussianNB()
    sk_nb.fit(X_train, y_train)
    y_pred = sk_nb.predict(X_test)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c='#3498db', label='类别0(训练)', alpha=0.6, edgecolors='k')
    ax1.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c='#e74c3c', label='类别1(训练)', alpha=0.6, edgecolors='k')
    ax1.set_xlabel('特征1')
    ax1.set_ylabel('特征2')
    ax1.set_title('朴素贝叶斯 - 训练数据')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    correct = y_pred == y_test
    ax2.scatter(X_test[correct, 0], X_test[correct, 1], c='#2ecc71', marker='o', label='分类正确', alpha=0.7, edgecolors='k')
    ax2.scatter(X_test[~correct, 0], X_test[~correct, 1], c='#e74c3c', marker='x', s=100, label='分类错误', linewidths=2)
    ax2.set_xlabel('特征1')
    ax2.set_ylabel('特征2')
    ax2.set_title('朴素贝叶斯 - 分类结果散点图')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
