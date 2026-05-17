"""逻辑回归文本分类模块

使用逻辑回归（Logistic Regression）对文本特征进行二分类，
包含StandardScaler标准化预处理，训练模型并评估准确率、精确率、召回率与AUC。
"""
import os
import h5py
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    with h5py.File(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'train_onehot.hdf5'), 'r') as train_file:
        X_train = train_file['X_train'][:]
        y_train = train_file['y_train'][:].reshape(-1)

    with h5py.File(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'test_onehot.hdf5'), 'r') as test_file:
        X_test = test_file['X_test'][:]
        y_test = test_file['y_test'][:].reshape(-1)

    X_train = X_train.reshape(-1, X_train.shape[-2] * X_train.shape[-1])
    X_test = X_test.reshape(-1, X_test.shape[-2] * X_test.shape[-1])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    logistic_regression_model = LogisticRegression()

    logistic_regression_model.fit(X_train, y_train)

    y_pred = logistic_regression_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    y_prob = logistic_regression_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy: {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall: {rec:.3f}")
    print(f"AUC: {auc:.3f}")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label='Logistic Regression')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.show()

    print("Done!")
