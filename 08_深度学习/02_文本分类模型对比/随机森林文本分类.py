import os
import h5py
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt


# 本项目完成了数据加载、预处理、模型构建、训练、评估到结果可视化的一整套机器学习流程，主要用于二分类问题。
# 1. 数据加载
# 目的：从HDF5文件中读取训练集和测试集数据。
# 实现：使用h5py.File以读取模式打开指定路径的HDF5文件，通过键（如'X_train'和'y_train'）访问并加载数据到内存中。此操作进行了两次，分别加载训练集和测试集。
# 2. 数据预处理
# 展平数据：由于原始数据可能是多维的（例如，图像数据），通过.reshape(-1, ...)操作将其转换为一维数组，以便输入到模型中。这一步确保了模型能够处理这些数据。
# 3. 构建随机森林模型
# 模型定义：使用sklearn.ensemble.RandomForestClassifier创建了一个随机森林分类器实例，设置了两个参数：
# n_estimators=100：表示随机森林中的树的数量为100。
# random_state=42：设定随机种子，确保实验结果可复现。
# 4. 模型训练
# fit方法：利用训练数据集X_train_flat和对应的标签y_train，调用fit方法训练随机森林模型。这一过程涉及在训练数据上构建多个决策树，并汇总它们的预测结果。
# 5. 模型预测与评估
# 预测：使用predict方法对测试集X_test_flat进行分类预测，得到预测标签y_pred。
# 性能指标计算：
# 准确率（Accuracy）：通过accuracy_score计算预测正确率。
# 精确率（Precision）：通过precision_score衡量预测为正例中实际为正例的比例。
# 召回率（Recall）：通过recall_score衡量所有实际正例中被正确识别的比例。
# AUC-ROC：通过roc_auc_score和roc_curve计算曲线下面积（Area Under the Curve），评估模型区分正负样本的能力。同时，绘制ROC曲线以直观展示。
# 6. 结果可视化
# ROC曲线：使用matplotlib.pyplot绘制ROC曲线，展示真阳性率（TPR）与假阳性率（FPR）之间的关系，用于评估分类器的诊断能力。
# 7. 输出与标识
# 输出结果：打印出模型的主要评价指标。


# 加载数据
with h5py.File(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'train_onehot.hdf5'), 'r') as train_file:
    X_train = train_file['X_train'][:]  # 获取特征数据集，并使用切片操作[:]进行复制，赋值给变量X_train。
    y_train = train_file['y_train'][:].reshape(-1)  # 重塑为一维数组，赋值给变量y_train

with h5py.File(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'test_onehot.hdf5'), 'r') as test_file:
    X_test = test_file['X_test'][:]
    y_test = test_file['y_test'][:].reshape(-1)

# 展平数据
X_train_flat = X_train.reshape(-1, X_train.shape[-2] * X_train.shape[-1])
X_test_flat = X_test.reshape(-1, X_test.shape[-2] * X_test.shape[-1])

# 构建随机森林模型
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练模型
random_forest_model.fit(X_train_flat, y_train)

# 预测
y_pred = random_forest_model.predict(X_test_flat)

# 计算评价指标
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
y_prob = random_forest_model.predict_proba(X_test_flat)[:, 1]
auc = roc_auc_score(y_test, y_prob)

print(f"Accuracy: {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall: {rec:.3f}")
print(f"AUC: {auc:.3f}")

# 绘制ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label='Random Forest')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.show()

print("Done!")
print("20211113492 陈文聪 \n")


