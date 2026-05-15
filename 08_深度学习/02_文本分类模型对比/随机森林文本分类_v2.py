import os
import h5py
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# # 数据文件路径
# train_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'train_onehot.hdf5')
# test_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'test_onehot.hdf5')


# # 数据文件路径
train_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'train_onehot.hdf5')
test_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'test_onehot.hdf5')

# 加载数据
with h5py.File(train_file_path, 'r') as train_file:
    X_train = train_file['X_train'][:]
    y_train = train_file['y_train'][:].reshape(-1)

with h5py.File(test_file_path, 'r') as test_file:
    X_test = test_file['X_test'][:]
    y_test = test_file['y_test'][:].reshape(-1)

# 检查并计算特征数量
def calculate_n_features(data):
    if len(data.shape) == 2:  # 二维数据
        return data.shape[1]
    elif len(data.shape) == 3:  # 假定为三维数据
        return data.shape[1] * data.shape[2]
    else:
        raise ValueError("Unsupported data dimensionality.")

n_features_train = calculate_n_features(X_train)
n_features_test = calculate_n_features(X_test)

# 展平数据
X_train_flat = X_train.reshape(X_train.shape[0], n_features_train)
X_test_flat = X_test.reshape(X_test.shape[0], n_features_test)

print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
print("X_train_flat shape:", X_train_flat.shape)
print("X_test_flat shape:", X_test_flat.shape)

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

# print("Done!")
# print("20211113492 陈文聪 \n")





