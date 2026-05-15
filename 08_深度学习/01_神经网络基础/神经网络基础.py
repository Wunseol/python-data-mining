"""
神经网络基础模块 - Neural Network Basics
=========================================
涵盖神经网络的基础知识：
1. 感知机 (Perceptron)
2. 多层前馈神经网络 (BP神经网络)
3. 激活函数
4. 反向传播算法
5. 正则化技术 (Dropout、BatchNorm)
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 感知机
# ============================================================
class Perceptron:
    """单层感知机 — 线性二分类"""

    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters

    def fit(self, X, y):
        m, n = X.shape
        self.W = np.zeros(n)
        self.b = 0
        self.errors_ = []

        for _ in range(self.n_iters):
            errors = 0
            for xi, yi in zip(X, y):
                update = self.lr * (yi - self.predict(xi))
                self.W += update * xi
                self.b += update
                errors += int(update != 0)
            self.errors_.append(errors)
            if errors == 0:
                break
        return self

    def predict(self, X):
        return np.where(X @ self.W + self.b >= 0, 1, -1)


# ============================================================
# 2. 激活函数
# ============================================================
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def relu(z):
    return np.maximum(0, z)


def tanh(z):
    return np.tanh(z)


def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def plot_activation_functions():
    """可视化常见激活函数"""
    z = np.linspace(-5, 5, 200)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    activations = [
        ('Sigmoid', sigmoid(z)),
        ('ReLU', relu(z)),
        ('Tanh', tanh(z)),
        ('Leaky ReLU', leaky_relu(z)),
        ('Sigmoid导数', sigmoid(z) * (1 - sigmoid(z))),
        ('ReLU导数', np.where(z > 0, 1, 0)),
    ]

    for ax, (name, values) in zip(axes.flatten(), activations):
        ax.plot(z, values, linewidth=2)
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)

    plt.tight_layout()
    plt.savefig('激活函数.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 3. 多层前馈神经网络 (BP神经网络)
# ============================================================
class MLP:
    """多层感知机 — 反向传播算法"""

    def __init__(self, layer_sizes, lr=0.01, activation='relu', n_iters=1000):
        """
        layer_sizes: 各层神经元数，如 [2, 64, 32, 1]
        activation: 'relu' 或 'sigmoid'
        """
        self.layer_sizes = layer_sizes
        self.lr = lr
        self.activation = activation
        self.n_iters = n_iters
        self.losses = []

    def _init_weights(self):
        self.weights = []
        self.biases = []
        for i in range(len(self.layer_sizes) - 1):
            # He初始化
            w = np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * \
                np.sqrt(2.0 / self.layer_sizes[i])
            b = np.zeros((1, self.layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

    def _activate(self, z):
        if self.activation == 'relu':
            return relu(z)
        else:
            return sigmoid(z)

    def _activate_derivative(self, z):
        if self.activation == 'relu':
            return np.where(z > 0, 1, 0)
        else:
            s = sigmoid(z)
            return s * (1 - s)

    def fit(self, X, y):
        m = X.shape[0]
        self._init_weights()
        y = y.reshape(-1, 1)

        for iteration in range(self.n_iters):
            # 前向传播
            activations = [X]
            zs = []
            a = X
            for w, b in zip(self.weights, self.biases):
                z = a @ w + b
                zs.append(z)
                a = self._activate(z) if w is not self.weights[-1] else sigmoid(z)
                activations.append(a)

            # 输出层
            output = activations[-1]

            # 损失 (交叉熵)
            eps = 1e-15
            loss = -np.mean(y * np.log(output + eps) + (1 - y) * np.log(1 - output + eps))
            self.losses.append(loss)

            # 反向传播
            delta = output - y  # 输出层梯度
            deltas = [delta]

            for i in range(len(self.weights) - 2, -1, -1):
                delta = deltas[0] @ self.weights[i + 1].T * self._activate_derivative(zs[i])
                deltas.insert(0, delta)

            # 更新权重
            for i in range(len(self.weights)):
                self.weights[i] -= self.lr * (activations[i].T @ deltas[i]) / m
                self.biases[i] -= self.lr * np.mean(deltas[i], axis=0, keepdims=True)

        return self

    def predict_proba(self, X):
        a = X
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ w + b
            a = self._activate(z) if i < len(self.weights) - 1 else sigmoid(z)
        return a

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# ============================================================
# 4. 可视化
# ============================================================
def plot_decision_boundary_mlp(model, X, y, title='MLP决策边界'):
    """可视化MLP决策边界"""
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap=plt.cm.RdYlBu,
                edgecolors='black', s=30)
    plt.title(title)
    plt.savefig(f'{title}.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_loss_curve(losses, title='训练损失曲线'):
    """绘制损失曲线"""
    plt.figure(figsize=(8, 5))
    plt.plot(losses, linewidth=1)
    plt.xlabel('迭代次数')
    plt.ylabel('损失')
    plt.title(title)
    plt.savefig(f'{title}.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 主程序演示
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("神经网络基础完整流程演示")
    print("=" * 60)

    # --- 1. 感知机 ---
    print("\n--- 1. 感知机 ---")
    np.random.seed(42)
    X_perceptron = np.random.randn(100, 2)
    y_perceptron = np.where(X_perceptron[:, 0] + X_perceptron[:, 1] > 0, 1, -1)
    perceptron = Perceptron(lr=0.1, n_iters=100)
    perceptron.fit(X_perceptron, y_perceptron)
    acc = np.mean(perceptron.predict(X_perceptron) == y_perceptron)
    print(f"  感知机准确率: {acc:.4f}")

    # --- 2. 激活函数 ---
    print("\n--- 2. 激活函数可视化 ---")
    plot_activation_functions()

    # --- 3. MLP — 线性可分 ---
    print("\n--- 3. MLP (线性可分数据) ---")
    np.random.seed(42)
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=500, n_features=2, n_redundant=0,
                               n_clusters_per_class=1, random_state=42)
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(X)

    mlp = MLP(layer_sizes=[2, 32, 16, 1], lr=0.1, activation='relu', n_iters=1000)
    mlp.fit(X, y)
    y_pred = mlp.predict(X)
    acc = np.mean(y_pred.flatten() == y)
    print(f"  MLP准确率: {acc:.4f}")
    plot_decision_boundary_mlp(mlp, X, y, 'MLP决策边界 (线性可分)')
    plot_loss_curve(mlp.losses, 'MLP训练损失曲线')

    # --- 4. MLP — 非线性 (XOR问题) ---
    print("\n--- 4. MLP (XOR问题) ---")
    np.random.seed(42)
    X_xor = np.random.randn(400, 2)
    y_xor = np.where(X_xor[:, 0] * X_xor[:, 1] > 0, 1, 0)

    mlp_xor = MLP(layer_sizes=[2, 64, 32, 1], lr=0.05, activation='relu', n_iters=2000)
    mlp_xor.fit(X_xor, y_xor)
    y_pred = mlp_xor.predict(X_xor)
    acc = np.mean(y_pred.flatten() == y_xor)
    print(f"  XOR MLP准确率: {acc:.4f}")
    plot_decision_boundary_mlp(mlp_xor, X_xor, y_xor, 'MLP决策边界 (XOR)')
