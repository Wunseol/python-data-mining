"""
联邦学习与隐私保护 - Federated Learning and Privacy Preservation
================================================================
系统化覆盖联邦学习与隐私保护核心知识：

1. 联邦学习概述：横向/纵向/联邦迁移学习
2. FedAvg 算法手动实现：模拟多客户端联邦训练
3. Non-IID 数据划分：Dirichlet 分布非独立同分布数据划分
4. 差分隐私基础：Laplace/Gaussian 机制
5. DP-SGD 简化实现：差分隐私随机梯度下降
6. 可视化分析：收敛曲线、隐私-效用权衡
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import setup_chinese_font

setup_chinese_font()

# ============================================================================
# 联邦学习概述
# ============================================================================

def print_federated_learning_overview():
    print("=" * 60)
    print("联邦学习概述")
    print("=" * 60)

    print("\n横向联邦学习 (Horizontal FL):")
    print("  - 相同特征空间, 不同样本")
    print("  - 各客户端拥有相同类型的特征，但数据样本不同")
    print("  - 例: 不同地区的银行，业务相同但客户不同")

    print("\n纵向联邦学习 (Vertical FL):")
    print("  - 相同样本, 不同特征")
    print("  - 各客户端拥有同一样本的不同特征视图")
    print("  - 例: 同一用户在银行和电商的不同数据")

    print("\n联邦迁移学习 (Federated Transfer Learning):")
    print("  - 不同样本, 不同特征")
    print("  - 跨领域知识迁移，解决数据与特征均不同的问题")
    print("  - 例: 不同国家不同业务的联合建模")

    print("\n核心挑战:")
    print("  1. 通信效率: 减少客户端与服务器之间的通信轮次和数据量")
    print("  2. 数据异构 (Non-IID): 各客户端数据分布差异大")
    print("  3. 隐私保护: 防止从模型更新中推断原始数据")


# ============================================================================
# 工具函数: Sigmoid 与交叉熵
# ============================================================================

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def binary_cross_entropy(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# ============================================================================
# FedAvg 算法手动实现
# ============================================================================

class FedAvgManual:
    """FedAvg 联邦平均算法手动实现"""

    def __init__(self, n_clients=5, n_rounds=10, local_epochs=5, lr=0.01):
        self.n_clients = n_clients
        self.n_rounds = n_rounds
        self.local_epochs = local_epochs
        self.lr = lr
        self.global_weights = None
        self.global_bias = None
        self.round_accuracies = []

    def _init_model(self, n_features):
        rng = np.random.RandomState(42)
        self.global_weights = rng.randn(n_features) * 0.01
        self.global_bias = 0.0

    def _local_train(self, X, y):
        w = self.global_weights.copy()
        b = self.global_bias
        for _ in range(self.local_epochs):
            z = X @ w + b
            pred = sigmoid(z)
            error = pred - y
            grad_w = X.T @ error / len(y)
            grad_b = np.mean(error)
            w -= self.lr * grad_w
            b -= self.lr * grad_b
        return w, b, len(y)

    def fit(self, X_list, y_list):
        n_features = X_list[0].shape[1]
        self._init_model(n_features)
        self.round_accuracies = []

        X_test = self._X_test
        y_test = self._y_test

        for rnd in range(self.n_rounds):
            local_results = []
            for i in range(self.n_clients):
                w, b, n = self._local_train(X_list[i], y_list[i])
                local_results.append((w, b, n))

            total_n = sum(r[2] for r in local_results)
            self.global_weights = sum(r[0] * r[2] for r in local_results) / total_n
            self.global_bias = sum(r[1] * r[2] for r in local_results) / total_n

            y_pred = self.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            self.round_accuracies.append(acc)
            print(f"  Round {rnd + 1:3d}/{self.n_rounds} - 全局模型准确率: {acc:.4f}")

    def predict(self, X):
        z = X @ self.global_weights + self.global_bias
        return (sigmoid(z) >= 0.5).astype(int)

    def set_test_data(self, X_test, y_test):
        self._X_test = X_test
        self._y_test = y_test


# ============================================================================
# Non-IID 数据划分
# ============================================================================

def split_noniid(X, y, n_clients, alpha=0.5):
    rng = np.random.RandomState(42)
    n_classes = len(np.unique(y))
    client_indices = [[] for _ in range(n_clients)]

    for k in range(n_classes):
        idx_k = np.where(y == k)[0]
        rng.shuffle(idx_k)
        proportions = rng.dirichlet(np.repeat(alpha, n_clients))
        proportions = (proportions * len(idx_k)).astype(int)
        diff = len(idx_k) - proportions.sum()
        proportions[0] += diff

        start = 0
        for i in range(n_clients):
            client_indices[i].extend(idx_k[start:start + proportions[i]])
            start += proportions[i]

    X_list = [X[indices] for indices in client_indices]
    y_list = [y[indices] for indices in client_indices]
    return X_list, y_list


def split_iid(X, y, n_clients):
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(X))
    split_size = len(X) // n_clients
    X_list = []
    y_list = []
    for i in range(n_clients):
        start = i * split_size
        end = start + split_size if i < n_clients - 1 else len(X)
        idx = indices[start:end]
        X_list.append(X[idx])
        y_list.append(y[idx])
    return X_list, y_list


def visualize_label_distribution(X_list_iid, y_list_iid, X_list_noniid, y_list_noniid):
    n_clients = len(X_list_iid)
    n_classes = len(np.unique(np.concatenate(y_list_iid)))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, y_list, title in [
        (axes[0], y_list_iid, 'IID 数据分布'),
        (axes[1], y_list_noniid, 'Non-IID 数据分布 (Dirichlet α=0.5)')
    ]:
        counts = np.zeros((n_clients, n_classes))
        for i in range(n_clients):
            for c in range(n_classes):
                counts[i, c] = np.sum(y_list[i] == c)
        x_pos = np.arange(n_clients)
        width = 0.8 / n_classes
        for c in range(n_classes):
            ax.bar(x_pos + c * width, counts[:, c], width, label=f'类别 {c}')
        ax.set_xlabel('客户端')
        ax.set_ylabel('样本数')
        ax.set_title(title)
        ax.set_xticks(x_pos + width * (n_classes - 1) / 2)
        ax.set_xticklabels([f'Client {i}' for i in range(n_clients)])
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'label_distribution.png'), dpi=150)
    plt.close()
    print("  标签分布图已保存: label_distribution.png")


# ============================================================================
# 差分隐私基础
# ============================================================================

def laplace_mechanism(value, sensitivity, epsilon):
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale, size=np.shape(value))
    return value + noise


def gaussian_mechanism(value, sensitivity, epsilon, delta):
    sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    noise = np.random.normal(0, sigma, size=np.shape(value))
    return value + noise


def demonstrate_privacy_utility_tradeoff():
    rng = np.random.RandomState(42)
    X, y = make_classification(
        n_samples=500, n_features=10, n_informative=5,
        n_redundant=0, n_classes=2, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    n_features = X_train.shape[1]
    epsilons = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
    n_repeats = 10
    laplace_accs = []
    gaussian_accs = []

    for eps in epsilons:
        lap_runs = []
        gau_runs = []
        for _ in range(n_repeats):
            rng_iter = np.random.RandomState(42)
            w = rng_iter.randn(n_features) * 0.01
            b = 0.0
            lr = 0.01
            for epoch in range(50):
                z = X_train @ w + b
                pred = sigmoid(z)
                error = pred - y_train
                grad_w = X_train.T @ error / len(y_train)
                grad_b = np.mean(error)
                w -= lr * laplace_mechanism(grad_w, 1.0, eps)
                b -= lr * laplace_mechanism(np.array([grad_b]), 1.0, eps)[0]

            y_pred = (sigmoid(X_test @ w + b) >= 0.5).astype(int)
            lap_runs.append(accuracy_score(y_test, y_pred))

            w = rng_iter.randn(n_features) * 0.01
            b = 0.0
            for epoch in range(50):
                z = X_train @ w + b
                pred = sigmoid(z)
                error = pred - y_train
                grad_w = X_train.T @ error / len(y_train)
                grad_b = np.mean(error)
                w -= lr * gaussian_mechanism(grad_w, 1.0, eps, 1e-5)
                b -= lr * gaussian_mechanism(np.array([grad_b]), 1.0, eps, 1e-5)[0]

            y_pred = (sigmoid(X_test @ w + b) >= 0.5).astype(int)
            gau_runs.append(accuracy_score(y_test, y_pred))

        laplace_accs.append(np.mean(lap_runs))
        gaussian_accs.append(np.mean(gau_runs))

    print("\n  Epsilon\tLaplace准确率\tGaussian准确率")
    print("  " + "-" * 48)
    for i, eps in enumerate(epsilons):
        print(f"  {eps:>8.1f}\t{laplace_accs[i]:.4f}\t\t{gaussian_accs[i]:.4f}")

    return epsilons, laplace_accs, gaussian_accs


def visualize_privacy_utility(epsilons, laplace_accs, gaussian_accs):
    plt.figure(figsize=(8, 5))
    plt.plot(epsilons, laplace_accs, 'o-', label='Laplace 机制')
    plt.plot(epsilons, gaussian_accs, 's-', label='Gaussian 机制')
    plt.xscale('log')
    plt.xlabel('隐私预算 ε (log scale)')
    plt.ylabel('准确率')
    plt.title('差分隐私: 隐私-效用权衡')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'privacy_utility_tradeoff.png'), dpi=150)
    plt.close()
    print("  隐私-效用权衡图已保存: privacy_utility_tradeoff.png")


# ============================================================================
# DP-SGD 简化实现
# ============================================================================

class DPSGDManual:
    """差分隐私随机梯度下降 (DP-SGD) 简化实现"""

    def __init__(self, lr=0.01, n_epochs=50, clip_norm=1.0, epsilon=1.0, delta=1e-5):
        self.lr = lr
        self.n_epochs = n_epochs
        self.clip_norm = clip_norm
        self.epsilon = epsilon
        self.delta = delta
        self.weights = None
        self.bias = None
        self.train_losses = []

    def _clip_gradient(self, grad, norm):
        grad_norm = np.linalg.norm(grad)
        if grad_norm > norm:
            grad = grad * (norm / grad_norm)
        return grad

    def _compute_noise_scale(self, n_samples):
        sigma = self.clip_norm * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        return sigma

    def fit(self, X, y):
        rng = np.random.RandomState(42)
        n_samples, n_features = X.shape
        self.weights = rng.randn(n_features) * 0.01
        self.bias = 0.0
        self.train_losses = []
        noise_scale = self._compute_noise_scale(n_samples)

        for epoch in range(self.n_epochs):
            z = X @ self.weights + self.bias
            pred = sigmoid(z)
            loss = binary_cross_entropy(y, pred)
            self.train_losses.append(loss)

            per_sample_grads_w = (pred - y)[:, np.newaxis] * X / n_samples
            per_sample_grads_b = (pred - y) / n_samples

            clipped_grads_w = np.array([
                self._clip_gradient(g, self.clip_norm) for g in per_sample_grads_w
            ])
            clipped_grads_b = np.array([
                self._clip_gradient(np.array([g]), self.clip_norm)[0] for g in per_sample_grads_b
            ])

            agg_grad_w = np.sum(clipped_grads_w, axis=0) / n_samples
            agg_grad_b = np.mean(clipped_grads_b)

            noise_w = np.random.normal(0, noise_scale / n_samples, n_features)
            noise_b = np.random.normal(0, noise_scale / n_samples)

            self.weights -= self.lr * (agg_grad_w + noise_w)
            self.bias -= self.lr * (agg_grad_b + noise_b)

        return self

    def predict(self, X):
        z = X @ self.weights + self.bias
        return (sigmoid(z) >= 0.5).astype(int)


class StandardSGDManual:
    """标准 SGD (无隐私保护) 用于对比"""

    def __init__(self, lr=0.01, n_epochs=50):
        self.lr = lr
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None
        self.train_losses = []

    def fit(self, X, y):
        rng = np.random.RandomState(42)
        n_samples, n_features = X.shape
        self.weights = rng.randn(n_features) * 0.01
        self.bias = 0.0
        self.train_losses = []

        for epoch in range(self.n_epochs):
            z = X @ self.weights + self.bias
            pred = sigmoid(z)
            loss = binary_cross_entropy(y, pred)
            self.train_losses.append(loss)

            error = pred - y
            grad_w = X.T @ error / n_samples
            grad_b = np.mean(error)

            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_b

        return self

    def predict(self, X):
        z = X @ self.weights + self.bias
        return (sigmoid(z) >= 0.5).astype(int)


def compare_dpsgd_vs_sgd(X_train, X_test, y_train, y_test):
    print("\n  DP-SGD vs 标准 SGD 对比:")
    print("  " + "-" * 50)

    sgd = StandardSGDManual(lr=0.01, n_epochs=50)
    sgd.fit(X_train, y_train)
    sgd_pred = sgd.predict(X_test)
    sgd_acc = accuracy_score(y_test, sgd_pred)
    print(f"  标准 SGD 准确率: {sgd_acc:.4f}")

    dp_results = {}
    for eps in [0.1, 1.0, 5.0, 10.0]:
        dp = DPSGDManual(lr=0.01, n_epochs=50, clip_norm=1.0, epsilon=eps, delta=1e-5)
        dp.fit(X_train, y_train)
        dp_pred = dp.predict(X_test)
        dp_acc = accuracy_score(y_test, dp_pred)
        dp_results[eps] = (dp_acc, dp.train_losses)
        print(f"  DP-SGD (ε={eps:>5.1f}) 准确率: {dp_acc:.4f}")

    return sgd.train_losses, dp_results, sgd_acc


def visualize_dpsgd_comparison(sgd_losses, dp_results, sgd_acc):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(sgd_losses, label=f'标准 SGD (Acc={sgd_acc:.4f})', linewidth=2)
    for eps, (acc, losses) in dp_results.items():
        axes[0].plot(losses, label=f'DP-SGD ε={eps} (Acc={acc:.4f})', linestyle='--')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('训练损失')
    axes[0].set_title('训练损失曲线: 标准 SGD vs DP-SGD')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    eps_list = list(dp_results.keys())
    accs = [dp_results[e][0] for e in eps_list]
    eps_list_with_sgd = [0] + eps_list
    accs_with_sgd = [sgd_acc] + accs
    colors = ['#2ecc71'] + ['#e74c3c'] * len(eps_list)
    axes[1].bar(range(len(eps_list_with_sgd)), accs_with_sgd, color=colors, alpha=0.8)
    axes[1].set_xticks(range(len(eps_list_with_sgd)))
    axes[1].set_xticklabels(['标准 SGD'] + [f'ε={e}' for e in eps_list])
    axes[1].set_ylabel('测试准确率')
    axes[1].set_title('准确率对比: 标准 SGD vs DP-SGD')
    axes[1].set_ylim(0, 1.05)
    for i, v in enumerate(accs_with_sgd):
        axes[1].text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'dpsgd_comparison.png'), dpi=150)
    plt.close()
    print("  DP-SGD 对比图已保存: dpsgd_comparison.png")


# ============================================================================
# FedAvg 收敛曲线可视化
# ============================================================================

def visualize_fedavg_convergence(iid_accuracies, noniid_accuracies):
    plt.figure(figsize=(8, 5))
    rounds = range(1, len(iid_accuracies) + 1)
    plt.plot(rounds, iid_accuracies, 'o-', label='IID 数据', linewidth=2)
    plt.plot(rounds, noniid_accuracies, 's-', label='Non-IID 数据 (α=0.5)', linewidth=2)
    plt.xlabel('通信轮次')
    plt.ylabel('全局模型准确率')
    plt.title('FedAvg 收敛曲线: IID vs Non-IID')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'fedavg_convergence.png'), dpi=150)
    plt.close()
    print("  FedAvg 收敛曲线已保存: fedavg_convergence.png")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == '__main__':
    print_federated_learning_overview()

    print("\n" + "=" * 60)
    print("FedAvg 算法手动实现")
    print("=" * 60)

    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10,
        n_redundant=2, n_classes=2, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    n_clients = 5

    print("\n--- IID 数据划分 ---")
    X_list_iid, y_list_iid = split_iid(X_train, y_train, n_clients)
    for i in range(n_clients):
        unique, counts = np.unique(y_list_iid[i], return_counts=True)
        print(f"  Client {i}: {len(y_list_iid[i])} 样本, 类别分布: {dict(zip(unique, counts))}")

    print("\n--- Non-IID 数据划分 (Dirichlet α=0.5) ---")
    X_list_noniid, y_list_noniid = split_noniid(X_train, y_train, n_clients, alpha=0.5)
    for i in range(n_clients):
        unique, counts = np.unique(y_list_noniid[i], return_counts=True)
        print(f"  Client {i}: {len(y_list_noniid[i])} 样本, 类别分布: {dict(zip(unique, counts))}")

    print("\n--- 标签分布可视化 ---")
    visualize_label_distribution(X_list_iid, y_list_iid, X_list_noniid, y_list_noniid)

    print("\n--- FedAvg 训练 (IID 数据) ---")
    fedavg_iid = FedAvgManual(n_clients=n_clients, n_rounds=20, local_epochs=5, lr=0.01)
    fedavg_iid.set_test_data(X_test, y_test)
    fedavg_iid.fit(X_list_iid, y_list_iid)
    iid_final_acc = accuracy_score(y_test, fedavg_iid.predict(X_test))
    print(f"  IID 最终测试准确率: {iid_final_acc:.4f}")

    print("\n--- FedAvg 训练 (Non-IID 数据) ---")
    fedavg_noniid = FedAvgManual(n_clients=n_clients, n_rounds=20, local_epochs=5, lr=0.01)
    fedavg_noniid.set_test_data(X_test, y_test)
    fedavg_noniid.fit(X_list_noniid, y_list_noniid)
    noniid_final_acc = accuracy_score(y_test, fedavg_noniid.predict(X_test))
    print(f"  Non-IID 最终测试准确率: {noniid_final_acc:.4f}")

    print("\n--- FedAvg 收敛曲线可视化 ---")
    visualize_fedavg_convergence(fedavg_iid.round_accuracies, fedavg_noniid.round_accuracies)

    print("\n" + "=" * 60)
    print("差分隐私基础")
    print("=" * 60)

    print("\n--- 差分隐私机制演示 ---")
    true_value = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    print(f"  原始值: {true_value}")
    for eps in [0.1, 1.0, 10.0]:
        noisy_lap = laplace_mechanism(true_value, sensitivity=1.0, epsilon=eps)
        noisy_gau = gaussian_mechanism(true_value, sensitivity=1.0, epsilon=eps, delta=1e-5)
        print(f"  ε={eps:>5.1f}: Laplace → {noisy_lap.round(4)}, Gaussian → {noisy_gau.round(4)}")

    print("\n--- 隐私-效用权衡分析 ---")
    epsilons, laplace_accs, gaussian_accs = demonstrate_privacy_utility_tradeoff()

    print("\n--- 隐私-效用权衡可视化 ---")
    visualize_privacy_utility(epsilons, laplace_accs, gaussian_accs)

    print("\n" + "=" * 60)
    print("DP-SGD 简化实现")
    print("=" * 60)

    X_dp, y_dp = make_classification(
        n_samples=500, n_features=10, n_informative=5,
        n_redundant=0, n_classes=2, random_state=42
    )
    X_tr, X_te, y_tr, y_te = train_test_split(X_dp, y_dp, test_size=0.3, random_state=42)

    sgd_losses, dp_results, sgd_acc = compare_dpsgd_vs_sgd(X_tr, X_te, y_tr, y_te)

    print("\n--- DP-SGD 对比可视化 ---")
    visualize_dpsgd_comparison(sgd_losses, dp_results, sgd_acc)

    print("\n" + "=" * 60)
    print("全部实验完成!")
    print("=" * 60)
