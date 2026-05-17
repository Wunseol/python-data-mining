"""SVM文本分类模块

使用支持向量机（SVM）对文本特征进行二分类，
通过GridSearchCV网格搜索调参，训练最优模型并评估分类性能。
"""
import os
import h5py
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
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

    svm_classifier = svm.SVC()

    param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}
    grid_search = GridSearchCV(svm_classifier, param_grid, cv=5, scoring='accuracy', verbose=2)
    grid_search.fit(X_train, y_train)

    best_svm_classifier = grid_search.best_estimator_
    print("Best SVM parameters:", grid_search.best_params_)

    best_svm_classifier.fit(X_train, y_train)

    y_pred = best_svm_classifier.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    y_prob = best_svm_classifier.decision_function(X_test)
    auc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy: {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall: {rec:.3f}")
    print(f"AUC: {auc:.3f}")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label='SVM')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.show()
