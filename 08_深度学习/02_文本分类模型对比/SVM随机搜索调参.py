import os
import h5py
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import uniform

# 加载数据
with h5py.File(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'train_onehot.hdf5'), 'r') as train_file:
    X_train = train_file['X_train'][:]
    y_train = train_file['y_train'][:].reshape(-1)

with h5py.File(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'test_onehot.hdf5'), 'r') as test_file:
    X_test = test_file['X_test'][:]
    y_test = test_file['y_test'][:].reshape(-1)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

# 修改数据预处理部分
X_train = X_train.reshape(-1, X_train.shape[-2] * X_train.shape[-1])  # 展平最后一个两个维度
X_test = X_test.reshape(-1, X_test.shape[-2] * X_test.shape[-1])  # 展平最后一个两个维度

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 构建SVM模型
svm_classifier = svm.SVC()

print("构建SVM模型")
print("开始训练")

# 使用随机搜索调整SVM参数
param_distributions = {
    'C': uniform(loc=0.01, scale=10),
    'kernel': ['linear', 'rbf'],
}

print("开始随机搜索")

random_search = RandomizedSearchCV(
    svm_classifier,
    param_distributions,
    # n_jobs=-1,  # 启用并行交叉验证
    n_iter=1,  # 调整此值以控制搜索次数
    cv=2,
    scoring='accuracy',
    random_state=42,
)
random_search.fit(X_train, y_train)

print("最佳参数：", random_search.best_params_)

best_svm_classifier = random_search.best_estimator_
print("Best SVM parameters:", random_search.best_params_)

print("Best SVM accuracy:", random_search.best_score_)

# 训练模型
best_svm_classifier.fit(X_train, y_train)

# 预测
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
