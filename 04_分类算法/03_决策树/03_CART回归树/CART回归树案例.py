"""
CART回归树 - 模型树案例
运行此脚本前，请确保 regTrees.py 在同一目录下
"""
import regTrees
import numpy as np

# 加载训练数据集
trainMat = regTrees.loadDataSet('bikeSpeedVsIq_train.txt')
# 加载测试数据集
testMat = regTrees.loadDataSet('bikeSpeedVsIq_test.txt')

# 使用训练数据创建回归树
myTree = regTrees.createTree(trainMat, ops=(1, 20))

# 对测试数据集进行预测
yHat = regTrees.createForeCast(myTree, testMat[:, 0])

# 计算预测值与实际值之间的相关系数
corr_coef = np.corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1]
print("回归树相关系数:", corr_coef)

# 使用模型树
myTree = regTrees.createTree(trainMat, regTrees.modelLeaf, regTrees.modelErr, (1, 20))
yHat = regTrees.createForeCast(myTree, testMat[:, 0], regTrees.modelTreeEval)
corr_coef = np.corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1]
print("模型树相关系数:", corr_coef)
