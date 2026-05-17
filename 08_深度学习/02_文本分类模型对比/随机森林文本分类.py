"""随机森林文本分类模块

使用随机森林（Random Forest）对文本特征进行二分类，
从HDF5文件加载One-Hot编码数据，训练模型并评估准确率、精确率、召回率与AUC。
"""
import os
import h5py
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt


if __name__ == '__main__':
    with h5py.File(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'train_onehot.hdf5'), 'r') as train_file:
        X_train = train_file['X_train'][:]
        y_train = train_file['y_train'][:].reshape(-1)

    with h5py.File(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'test_onehot.hdf5'), 'r') as test_file:
        X_test = test_file['X_test'][:]
        y_test = test_file['y_test'][:].reshape(-1)

    X_train_flat = X_train.reshape(-1, X_train.shape[-2] * X_train.shape[-1])
    X_test_flat = X_test.reshape(-1, X_test.shape[-2] * X_test.shape[-1])

    random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)

    random_forest_model.fit(X_train_flat, y_train)

    y_pred = random_forest_model.predict(X_test_flat)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    y_prob = random_forest_model.predict_proba(X_test_flat)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy: {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall: {rec:.3f}")
    print(f"AUC: {auc:.3f}")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label='Random Forest')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.show()

    print("Done!")
