"""
逻辑回归模块 - Logistic Regression
===================================
涵盖逻辑回归的核心内容：
1. 二分类逻辑回归 (手动实现)
2. 多分类逻辑回归 (Softmax)
3. 正则化逻辑回归
4. 评估指标与ROC曲线
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_iris, load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import StandardScaler, label_binarize

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 二分类逻辑回归 (手动实现)
# ============================================================
class LogisticRegressionGD:
    """逻辑回归 — 梯度下降实现"""

    def __init__(self, lr=0.01, n_iters=1000, penalty=None, alpha=0.1):
        self.lr = lr
        self.n_iters = n_iters
        self.penalty = penalty      # None, 'l1', 'l2'
        self.alpha = alpha          # 正则化强度
        self.losses = []

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        m, n = X.shape
        self.W = np.zeros(n)
        self.b = 0

        for _ in range(self.n_iters):
            z = X @ self.W + self.b
            y_pred = self._sigmoid(z)

            # 交叉熵损失
            eps = 1e-15
            loss = -np.mean(y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps))

            # 正则化项
            if self.penalty == 'l2':
                loss += self.alpha * np.sum(self.W ** 2) / 2
            elif self.penalty == 'l1':
                loss += self.alpha * np.sum(np.abs(self.W))

            self.losses.append(loss)

            # 梯度
            dW = (1 / m) * (X.T @ (y_pred - y))
            db = (1 / m) * np.sum(y_pred - y)

            if self.penalty == 'l2':
                dW += self.alpha * self.W / m
            elif self.penalty == 'l1':
                dW += self.alpha * np.sign(self.W) / m

            self.W -= self.lr * dW
            self.b -= self.lr * db

        return self

    def predict_proba(self, X):
        return self._sigmoid(X @ self.W + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# ============================================================
# 2. 多分类逻辑回归 (Softmax)
# ============================================================
class SoftmaxRegression:
    """多分类逻辑回归 — Softmax回归"""

    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.losses = []

    @staticmethod
    def _softmax(z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y):
        m, n = X.shape
        self.n_classes = len(np.unique(y))
        self.W = np.zeros((n, self.n_classes))
        self.b = np.zeros(self.n_classes)

        # One-hot编码
        y_onehot = np.eye(self.n_classes)[y]

        for _ in range(self.n_iters):
            z = X @ self.W + self.b
            y_pred = self._softmax(z)

            loss = -np.mean(np.sum(y_onehot * np.log(y_pred + 1e-15), axis=1))
            self.losses.append(loss)

            dW = (1 / m) * (X.T @ (y_pred - y_onehot))
            db = (1 / m) * np.sum(y_pred - y_onehot, axis=0)

            self.W -= self.lr * dW
            self.b -= self.lr * db

        return self

    def predict_proba(self, X):
        return self._softmax(X @ self.W + self.b)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


# ============================================================
# 3. 评估指标与ROC曲线
# ============================================================
def evaluate_binary(y_true, y_pred, y_prob=None):
    """二分类评估"""
    print(f"  准确率 (Accuracy):  {accuracy_score(y_true, y_pred):.4f}")
    print(f"  精确率 (Precision): {precision_score(y_true, y_pred):.4f}")
    print(f"  召回率 (Recall):    {recall_score(y_true, y_pred):.4f}")
    print(f"  F1分数:             {f1_score(y_true, y_pred):.4f}")
    if y_prob is not None:
        print(f"  AUC-ROC:            {roc_auc_score(y_true, y_prob):.4f}")
    print(f"\n混淆矩阵:\n{confusion_matrix(y_true, y_pred)}")


def plot_roc_curve(y_true, y_prob, title='ROC曲线'):
    """绘制ROC曲线"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC曲线 (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('假正率 (FPR)')
    plt.ylabel('真正率 (TPR)')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.savefig('ROC曲线.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_decision_boundary(model, X, y, title='决策边界'):
    """绘制二维决策边界"""
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black', s=30)
    plt.title(title)
    plt.savefig('决策边界.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 主程序演示
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("逻辑回归完整流程演示")
    print("=" * 60)

    # --- 1. 二分类 — 手动实现 ---
    print("\n--- 1. 二分类逻辑回归 (手动实现) ---")
    X, y = make_classification(n_samples=500, n_features=2, n_redundant=0,
                               n_clusters_per_class=1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegressionGD(lr=0.1, n_iters=500)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)
    evaluate_binary(y_test, y_pred, y_prob)
    plot_decision_boundary(model, X_test_s, y_test, '逻辑回归决策边界')

    # --- 2. sklearn逻辑回归 ---
    print("\n--- 2. sklearn逻辑回归 (乳腺癌数据集) ---")
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=5000, random_state=42)
    lr.fit(X_train_s, y_train)
    y_pred = lr.predict(X_test_s)
    y_prob = lr.predict_proba(X_test_s)[:, 1]
    evaluate_binary(y_test, y_pred, y_prob)
    plot_roc_curve(y_test, y_prob, '乳腺癌分类ROC曲线')

    # --- 3. 多分类 — Softmax ---
    print("\n--- 3. 多分类逻辑回归 (Iris数据集) ---")
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    softmax_model = SoftmaxRegression(lr=0.1, n_iters=1000)
    softmax_model.fit(X_train_s, y_train)
    y_pred = softmax_model.predict(X_test_s)
    print(f"  多分类准确率: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
