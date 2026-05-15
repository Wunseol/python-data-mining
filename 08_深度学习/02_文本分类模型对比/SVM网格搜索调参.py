import os
import h5py
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np


# 本项目使用支持向量机（SVM）对数据进行分类并进行全面的性能评估：

# 1. 数据加载
# 目的：从HDF5文件中加载训练集和测试集数据。
# 实现：使用h5py.File以读取模式打开指定路径下的train.hdf5和test.hdf5文件，从中读取特征矩阵X_train、X_test和标签向量y_train、y_test。
# 2. 数据预处理
# 展平数据：将多维特征数组展平为一维，以便于后续处理。例如，如果原始数据是图像，则将其从二维或三维结构转换为一维向量。
# 标准化：使用StandardScaler对特征进行标准化处理，确保所有特征具有相同的尺度，这对于基于距离的算法（如SVM）尤为重要。
# 3. 网格搜索与模型选择
# 减小数据集：为了加速网格搜索过程，仅使用训练集的10% (train_size=0.1) 进行参数调优。
# 参数网格定义：定义了一个参数网格，用于探索不同组合的SVM超参数，包括正则化参数C、核函数类型（这里固定为linear）以及gamma自动缩放。
# 网格搜索执行：利用GridSearchCV进行交叉验证（5折，cv=5），寻找最佳参数组合，同时考虑模型的准确性(scoring='accuracy')，并设置为高并发运行(n_jobs=-1)以加快速度。
# 最佳模型选取：根据网格搜索结果，选择具有最高交叉验证得分的SVM模型参数。
# 4. 模型训练与评估
# 模型训练：使用找到的最佳参数配置，在整个训练集上训练SVM模型。
# 预测与评估：在测试集上进行预测，然后计算并打印出模型的准确率、精确率、召回率和曲线下面积(AUC)等性能指标。
# 决策函数与ROC曲线：通过decision_function得到每个样本的预测概率，并据此计算ROC曲线，进一步可视化模型区分正负样本的能力。
# 5. 可视化
# ROC曲线绘制：使用matplotlib.pyplot绘制ROC曲线，展示真阳性率(TPR)与假阳性率(FPR)之间的关系，以评估分类器的诊断能力。

# 加载训练数据和测试数据
with h5py.File(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'train_onehot.hdf5'), 'r') as train_file:
    X_train = train_file['X_train'][:]
    y_train = train_file['y_train'][:].reshape(-1)

with h5py.File(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'test_onehot.hdf5'), 'r') as test_file:
    X_test = test_file['X_test'][:]
    y_test = test_file['y_test'][:].reshape(-1)

# 将数据展平并标准化，以便进行机器学习模型的训练
X_train = X_train.reshape(-1, X_train.shape[-2] * X_train.shape[-1])
X_test = X_test.reshape(-1, X_test.shape[-2] * X_test.shape[-1])

# StandardScaler 是来自scikit-learn库的一个类，用于数据的标准化预处理。标准化处理的目的是将特征的值转换到同一尺度上，这样可以消除特征尺度不同对模型性能的影响。
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)# 将每个特征减去其均值后再除以其标准差，使得处理后的数据具有0均值和单位方差的特性。
X_test = scaler.transform(X_test)# 对测试集（X_test）进行相同的标准化变换，确保训练和测试数据在相同的尺度上，这对于保证模型评估的公平性至关重要。

# 从全部训练数据中抽取10%的数据用于网格搜索调参
X_train_small, _, y_train_small, _ = train_test_split(X_train, y_train, train_size=0.1, stratify=y_train)

# 初始化SVM分类器，并使用网格搜索优化参数
svm_classifier = svm.SVC()

# 使用网格搜索调整SVM参数
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear'], 'gamma': ['scale']}# 用于指定要搜索的参数及其候选值。
grid_search = GridSearchCV(svm_classifier, param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)# GridSearchCV 是来自scikit-learn库的一个类，用于执行网格搜索交叉验证。
grid_search.fit(X_train_small, y_train_small)# 使用网格搜索进行交叉验证，寻找最佳参数组合

# 输出最佳参数组合，并使用全部训练数据训练最佳SVM分类器
best_svm_classifier = grid_search.best_estimator_
print("Best SVM parameters:", grid_search.best_params_)

# 训练模型
best_svm_classifier.fit(X_train, y_train)

# 使用测试数据进行预测，并计算准确性、精确率、召回率和AUC值
y_pred = best_svm_classifier.predict(X_test)

# 计算评价指标
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
y_prob = best_svm_classifier.decision_function(X_test)
auc = roc_auc_score(y_test, y_prob)

print(f"Accuracy: {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall: {rec:.3f}")
print(f"AUC: {auc:.3f}")

# 绘制ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label='SVM')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.show()

print("Done!")
print("20211113492 陈文聪 \n")
