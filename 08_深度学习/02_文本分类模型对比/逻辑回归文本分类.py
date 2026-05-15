import os
import h5py
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np

# 本项目应用机器学习中的逻辑回归模型对数据进行分类，并评估其性能：
# 代码结构：
# 1. 数据加载
# 目的：从HDF5文件格式中加载训练集和测试集数据。
# 实现：使用h5py库打开指定路径下的train.hdf5和test.hdf5文件，分别读取X_train、y_train、X_test、y_test数据。
# 2. 数据预处理
# 展平数据：由于原始数据可能为多维（如图像数据），通过.reshape(-1, ...)将其转换为一维数组，以便于模型处理。
# 标准化：利用sklearn.preprocessing.StandardScaler对特征进行标准化处理，减去均值并除以标准差，确保所有特征在同一尺度上，这对于基于距离的模型（如逻辑回归）尤为重要。
# 3. 模型构建
# 逻辑回归模型：使用sklearn.linear_model.LogisticRegression创建一个逻辑回归模型对象，这是一种广泛应用于二分类问题的线性模型。
# 4. 模型训练
# fit方法：通过调用逻辑回归模型的fit方法，使用标准化后的训练数据X_train和标签y_train来训练模型。
# 5. 预测与评估
# 预测：利用训练好的模型对测试集X_test进行预测，得到预测类别y_pred。
# 性能指标计算：
# 准确率(accuracy_score)：正确预测的样本数占总样本的比例。
# 精确率(precision_score)：在被预测为正类别的样本中，真正是正类别的比例。
# 召回率(recall_score)：在实际为正类别的样本中，被正确预测为正类别的比例。
# AUC值(roc_auc_score)：ROC曲线下的面积，衡量分类器的总体性能。
# 概率预测与AUC计算：使用predict_proba得到每个测试样本属于正类的概率，进而计算AUC值。
# 6. ROC曲线绘制
# ROC曲线：通过roc_curve函数计算假正率(fpr)和真正率(tpr)，然后使用matplotlib.pyplot绘制ROC曲线，展示分类器的诊断能力。
# 7. 输出与标注
# 结果输出：打印出模型的主要评价指标。
# 作者信息：最后输出执行脚本的学生姓名和学号作为标识。


# 加载数据
with h5py.File(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'train_onehot.hdf5'), 'r') as train_file:
    X_train = train_file['X_train'][:]
    y_train = train_file['y_train'][:].reshape(-1)

with h5py.File(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'test_onehot.hdf5'), 'r') as test_file:
    X_test = test_file['X_test'][:]
    y_test = test_file['y_test'][:].reshape(-1)

# 展平数据并标准化
X_train = X_train.reshape(-1, X_train.shape[-2] * X_train.shape[-1])
X_test = X_test.reshape(-1, X_test.shape[-2] * X_test.shape[-1])

# StandardScaler对象，它是一个用于数据标准化的预处理方法。
# 使用fit_transform()方法对训练集X_train进行标准化处理。该方法会计算训练集的均值和标准差，并将每个特征的值减去均值，再除以标准差，从而使得每个特征的均值为0，标准差为1。
# transform()方法对测试集X_test进行标准化处理。该方法会使用之前计算得到的均值和标准差对测试集进行相同的处理，以保持数据的一致性和可比性。
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 构建逻辑回归模型
logistic_regression_model = LogisticRegression()

# 训练模型
logistic_regression_model.fit(X_train, y_train)

# 预测
y_pred = logistic_regression_model.predict(X_test)

# 计算评价指标
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
y_prob = logistic_regression_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)

print(f"Accuracy: {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall: {rec:.3f}")
print(f"AUC: {auc:.3f}")

# 绘制ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label='Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.show()

print("Done!")
print("20211113492 陈文聪 \n")

