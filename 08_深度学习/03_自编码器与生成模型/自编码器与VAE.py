"""
自编码器与VAE模块 - Autoencoders & Variational Autoencoders
============================================================
涵盖自编码器与生成模型的核心概念与实现：
1. 自编码器(AE)手动实现（编码-解码、梯度下降训练、反向传播）
2. 去噪自编码器(DAE)（噪声注入、鲁棒特征学习）
3. 变分自编码器(VAE)手动实现（重参数化技巧、KL散度、生成采样）
4. 异常检测应用（重构误差作为异常分数、AE vs VAE对比）
5. 可视化分析（潜在空间、重构对比、异常分数分布、VAE生成样本）
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from utils import setup_chinese_font

setup_chinese_font()


# ============================================================
# 辅助函数
# ============================================================

def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def min_max_scale(X):
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1.0
    X_scaled = (X - X_min) / X_range
    return X_scaled, X_min, X_range


def min_max_inverse(X_scaled, X_min, X_range):
    return X_scaled * X_range + X_min


def he_init(fan_in, fan_out, random_state=None):
    rng = np.random.RandomState(random_state)
    return rng.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)


def clip_gradients(grads, max_norm=5.0):
    total_norm = 0.0
    for key in grads:
        total_norm += np.sum(grads[key] ** 2)
    total_norm = np.sqrt(total_norm)
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-8)
        for key in grads:
            grads[key] *= scale
    return grads


# ============================================================
# 1. 自编码器(AE)手动实现
# ============================================================

class AutoencoderManual:
    """
    自编码器手动实现（NumPy）

    架构: input_dim → hidden_dim(ReLU) → latent_dim → hidden_dim(ReLU) → input_dim(Sigmoid)
    训练: 小批量梯度下降 + MSE损失 + 手动反向传播
    """

    def __init__(self, input_dim, hidden_dim=16, latent_dim=2, random_state=42):
        rng = np.random.RandomState(random_state)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.W1 = he_init(input_dim, hidden_dim, random_state=rng.randint(0, 10000))
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = he_init(hidden_dim, latent_dim, random_state=rng.randint(0, 10000))
        self.b2 = np.zeros((1, latent_dim))
        self.W3 = he_init(latent_dim, hidden_dim, random_state=rng.randint(0, 10000))
        self.b3 = np.zeros((1, hidden_dim))
        self.W4 = he_init(hidden_dim, input_dim, random_state=rng.randint(0, 10000))
        self.b4 = np.zeros((1, input_dim))

        self.loss_history = []

    def _forward(self, X):
        z1 = X @ self.W1 + self.b1
        h1 = relu(z1)
        z2 = h1 @ self.W2 + self.b2
        z3 = z2 @ self.W3 + self.b3
        h2 = relu(z3)
        z4 = h2 @ self.W4 + self.b4
        x_hat = sigmoid(z4)

        cache = {
            'X': X, 'z1': z1, 'h1': h1, 'z2': z2,
            'z3': z3, 'h2': h2, 'z4': z4, 'x_hat': x_hat
        }
        return x_hat, cache

    def _backward(self, cache):
        X = cache['X']
        z1, h1, z2 = cache['z1'], cache['h1'], cache['z2']
        z3, h2, z4, x_hat = cache['z3'], cache['h2'], cache['z4'], cache['x_hat']

        N = X.shape[0]

        d_x_hat = 2.0 * (x_hat - X) / N
        d_z4 = d_x_hat * sigmoid_derivative(z4)
        d_W4 = h2.T @ d_z4
        d_b4 = np.sum(d_z4, axis=0, keepdims=True)

        d_h2 = d_z4 @ self.W4.T
        d_z3 = d_h2 * relu_derivative(z3)
        d_W3 = z2.T @ d_z3
        d_b3 = np.sum(d_z3, axis=0, keepdims=True)

        d_z2 = d_z3 @ self.W3.T
        d_W2 = h1.T @ d_z2
        d_b2 = np.sum(d_z2, axis=0, keepdims=True)

        d_h1 = d_z2 @ self.W2.T
        d_z1 = d_h1 * relu_derivative(z1)
        d_W1 = X.T @ d_z1
        d_b1 = np.sum(d_z1, axis=0, keepdims=True)

        grads = {
            'W1': d_W1, 'b1': d_b1, 'W2': d_W2, 'b2': d_b2,
            'W3': d_W3, 'b3': d_b3, 'W4': d_W4, 'b4': d_b4
        }
        return grads

    def fit(self, X, epochs=200, lr=0.001):
        self.loss_history = []
        for epoch in range(epochs):
            x_hat, cache = self._forward(X)
            loss = np.mean((X - x_hat) ** 2)
            self.loss_history.append(loss)

            grads = self._backward(cache)
            grads = clip_gradients(grads)

            self.W1 -= lr * grads['W1']
            self.b1 -= lr * grads['b1']
            self.W2 -= lr * grads['W2']
            self.b2 -= lr * grads['b2']
            self.W3 -= lr * grads['W3']
            self.b3 -= lr * grads['b3']
            self.W4 -= lr * grads['W4']
            self.b4 -= lr * grads['b4']

            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")
        return self

    def encode(self, X):
        z1 = X @ self.W1 + self.b1
        h1 = relu(z1)
        z2 = h1 @ self.W2 + self.b2
        return z2

    def decode(self, Z):
        z3 = Z @ self.W3 + self.b3
        h2 = relu(z3)
        z4 = h2 @ self.W4 + self.b4
        x_hat = sigmoid(z4)
        return x_hat

    def reconstruct(self, X):
        Z = self.encode(X)
        return self.decode(Z)


# ============================================================
# 2. 去噪自编码器(DAE)
# ============================================================

class DenoisingAutoencoder:
    """
    去噪自编码器手动实现（NumPy）

    训练时向输入添加高斯噪声，学习从损坏数据中恢复原始数据，
    从而获得更鲁棒的特征表示。架构与AE相同，但目标为干净数据。
    """

    def __init__(self, input_dim, hidden_dim=16, latent_dim=2, random_state=42):
        rng = np.random.RandomState(random_state)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.rng = rng

        self.W1 = he_init(input_dim, hidden_dim, random_state=rng.randint(0, 10000))
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = he_init(hidden_dim, latent_dim, random_state=rng.randint(0, 10000))
        self.b2 = np.zeros((1, latent_dim))
        self.W3 = he_init(latent_dim, hidden_dim, random_state=rng.randint(0, 10000))
        self.b3 = np.zeros((1, hidden_dim))
        self.W4 = he_init(hidden_dim, input_dim, random_state=rng.randint(0, 10000))
        self.b4 = np.zeros((1, input_dim))

        self.loss_history = []

    def _add_noise(self, X, noise_factor=0.3):
        noisy = X + noise_factor * self.rng.randn(*X.shape)
        return np.clip(noisy, 0.0, 1.0)

    def _forward(self, X):
        z1 = X @ self.W1 + self.b1
        h1 = relu(z1)
        z2 = h1 @ self.W2 + self.b2
        z3 = z2 @ self.W3 + self.b3
        h2 = relu(z3)
        z4 = h2 @ self.W4 + self.b4
        x_hat = sigmoid(z4)

        cache = {
            'X': X, 'z1': z1, 'h1': h1, 'z2': z2,
            'z3': z3, 'h2': h2, 'z4': z4, 'x_hat': x_hat
        }
        return x_hat, cache

    def _backward(self, cache, X_clean):
        X_noisy = cache['X']
        z1, h1, z2 = cache['z1'], cache['h1'], cache['z2']
        z3, h2, z4, x_hat = cache['z3'], cache['h2'], cache['z4'], cache['x_hat']

        N = X_clean.shape[0]

        d_x_hat = 2.0 * (x_hat - X_clean) / N
        d_z4 = d_x_hat * sigmoid_derivative(z4)
        d_W4 = h2.T @ d_z4
        d_b4 = np.sum(d_z4, axis=0, keepdims=True)

        d_h2 = d_z4 @ self.W4.T
        d_z3 = d_h2 * relu_derivative(z3)
        d_W3 = z2.T @ d_z3
        d_b3 = np.sum(d_z3, axis=0, keepdims=True)

        d_z2 = d_z3 @ self.W3.T
        d_W2 = h1.T @ d_z2
        d_b2 = np.sum(d_z2, axis=0, keepdims=True)

        d_h1 = d_z2 @ self.W2.T
        d_z1 = d_h1 * relu_derivative(z1)
        d_W1 = X_noisy.T @ d_z1
        d_b1 = np.sum(d_z1, axis=0, keepdims=True)

        grads = {
            'W1': d_W1, 'b1': d_b1, 'W2': d_W2, 'b2': d_b2,
            'W3': d_W3, 'b3': d_b3, 'W4': d_W4, 'b4': d_b4
        }
        return grads

    def fit(self, X, epochs=200, lr=0.001, noise_factor=0.3):
        self.loss_history = []
        for epoch in range(epochs):
            X_noisy = self._add_noise(X, noise_factor)
            x_hat, cache = self._forward(X_noisy)
            loss = np.mean((X - x_hat) ** 2)
            self.loss_history.append(loss)

            grads = self._backward(cache, X)
            grads = clip_gradients(grads)

            self.W1 -= lr * grads['W1']
            self.b1 -= lr * grads['b1']
            self.W2 -= lr * grads['W2']
            self.b2 -= lr * grads['b2']
            self.W3 -= lr * grads['W3']
            self.b3 -= lr * grads['b3']
            self.W4 -= lr * grads['W4']
            self.b4 -= lr * grads['b4']

            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")
        return self

    def encode(self, X):
        z1 = X @ self.W1 + self.b1
        h1 = relu(z1)
        z2 = h1 @ self.W2 + self.b2
        return z2

    def decode(self, Z):
        z3 = Z @ self.W3 + self.b3
        h2 = relu(z3)
        z4 = h2 @ self.W4 + self.b4
        x_hat = sigmoid(z4)
        return x_hat

    def reconstruct(self, X):
        Z = self.encode(X)
        return self.decode(Z)


# ============================================================
# 3. 变分自编码器(VAE)手动实现
# ============================================================

class VAEManual:
    """
    变分自编码器手动实现（NumPy）

    编码器输出均值和对数方差两组参数，使用重参数化技巧实现可微采样，
    损失函数 = MSE重构损失 + KL散度正则化项。
    重参数化: z = mu + exp(0.5 * log_var) * epsilon, epsilon ~ N(0, I)
    KL散度: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    """

    def __init__(self, input_dim, hidden_dim=16, latent_dim=2, random_state=42):
        rng = np.random.RandomState(random_state)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.rng = rng

        self.W1 = he_init(input_dim, hidden_dim, random_state=rng.randint(0, 10000))
        self.b1 = np.zeros((1, hidden_dim))
        self.W_mu = he_init(hidden_dim, latent_dim, random_state=rng.randint(0, 10000))
        self.b_mu = np.zeros((1, latent_dim))
        self.W_logvar = he_init(hidden_dim, latent_dim, random_state=rng.randint(0, 10000))
        self.b_logvar = np.zeros((1, latent_dim))
        self.W3 = he_init(latent_dim, hidden_dim, random_state=rng.randint(0, 10000))
        self.b3 = np.zeros((1, hidden_dim))
        self.W4 = he_init(hidden_dim, input_dim, random_state=rng.randint(0, 10000))
        self.b4 = np.zeros((1, input_dim))

        self.loss_history = []
        self.kl_history = []

    def _forward(self, X):
        z1 = X @ self.W1 + self.b1
        h1 = relu(z1)
        mu = h1 @ self.W_mu + self.b_mu
        log_var = h1 @ self.W_logvar + self.b_logvar

        epsilon = self.rng.randn(X.shape[0], self.latent_dim)
        z = mu + np.exp(0.5 * log_var) * epsilon

        z3 = z @ self.W3 + self.b3
        h2 = relu(z3)
        z4 = h2 @ self.W4 + self.b4
        x_hat = sigmoid(z4)

        cache = {
            'X': X, 'z1': z1, 'h1': h1,
            'mu': mu, 'log_var': log_var, 'epsilon': epsilon, 'z': z,
            'z3': z3, 'h2': h2, 'z4': z4, 'x_hat': x_hat
        }
        return x_hat, cache

    def _compute_loss(self, X, x_hat, mu, log_var):
        mse = np.mean((X - x_hat) ** 2)
        kl = -0.5 * np.mean(np.sum(1 + log_var - mu ** 2 - np.exp(log_var), axis=1))
        return mse + kl, mse, kl

    def _backward(self, cache):
        X = cache['X']
        z1, h1 = cache['z1'], cache['h1']
        mu, log_var, epsilon, z = cache['mu'], cache['log_var'], cache['epsilon'], cache['z']
        z3, h2, z4, x_hat = cache['z3'], cache['h2'], cache['z4'], cache['x_hat']

        N = X.shape[0]

        d_x_hat = 2.0 * (x_hat - X) / N
        d_z4 = d_x_hat * sigmoid_derivative(z4)
        d_W4 = h2.T @ d_z4
        d_b4 = np.sum(d_z4, axis=0, keepdims=True)

        d_h2 = d_z4 @ self.W4.T
        d_z3 = d_h2 * relu_derivative(z3)
        d_W3 = z.T @ d_z3
        d_b3 = np.sum(d_z3, axis=0, keepdims=True)

        d_z = d_z3 @ self.W3.T

        d_mu = d_z + mu / N
        d_log_var = (d_z * 0.5 * np.exp(0.5 * log_var) * epsilon
                     + (-0.5 + 0.5 * np.exp(log_var)) / N)

        d_W_mu = h1.T @ d_mu
        d_b_mu = np.sum(d_mu, axis=0, keepdims=True)
        d_W_logvar = h1.T @ d_log_var
        d_b_logvar = np.sum(d_log_var, axis=0, keepdims=True)

        d_h1 = d_mu @ self.W_mu.T + d_log_var @ self.W_logvar.T
        d_z1 = d_h1 * relu_derivative(z1)
        d_W1 = X.T @ d_z1
        d_b1 = np.sum(d_z1, axis=0, keepdims=True)

        grads = {
            'W1': d_W1, 'b1': d_b1,
            'W_mu': d_W_mu, 'b_mu': d_b_mu,
            'W_logvar': d_W_logvar, 'b_logvar': d_b_logvar,
            'W3': d_W3, 'b3': d_b3,
            'W4': d_W4, 'b4': d_b4
        }
        return grads

    def fit(self, X, epochs=200, lr=0.001):
        self.loss_history = []
        self.kl_history = []
        for epoch in range(epochs):
            x_hat, cache = self._forward(X)
            total_loss, mse, kl = self._compute_loss(
                X, x_hat, cache['mu'], cache['log_var']
            )
            self.loss_history.append(total_loss)
            self.kl_history.append(kl)

            grads = self._backward(cache)
            grads = clip_gradients(grads)

            self.W1 -= lr * grads['W1']
            self.b1 -= lr * grads['b1']
            self.W_mu -= lr * grads['W_mu']
            self.b_mu -= lr * grads['b_mu']
            self.W_logvar -= lr * grads['W_logvar']
            self.b_logvar -= lr * grads['b_logvar']
            self.W3 -= lr * grads['W3']
            self.b3 -= lr * grads['b3']
            self.W4 -= lr * grads['W4']
            self.b4 -= lr * grads['b4']

            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss:.6f}, "
                      f"MSE: {mse:.6f}, KL: {kl:.6f}")
        return self

    def encode(self, X):
        z1 = X @ self.W1 + self.b1
        h1 = relu(z1)
        mu = h1 @ self.W_mu + self.b_mu
        log_var = h1 @ self.W_logvar + self.b_logvar
        return mu, log_var

    def decode(self, Z):
        z3 = Z @ self.W3 + self.b3
        h2 = relu(z3)
        z4 = h2 @ self.W4 + self.b4
        x_hat = sigmoid(z4)
        return x_hat

    def reconstruct(self, X):
        mu, _ = self.encode(X)
        return self.decode(mu)

    def sample(self, n_samples=100):
        z = self.rng.randn(n_samples, self.latent_dim)
        return self.decode(z)


# ============================================================
# 4. 应用：异常检测
# ============================================================

def create_anomaly_data(n_samples=300, n_features=10, n_outliers=15, random_state=42):
    """
    生成含异常点的合成数据

    使用 make_blobs 生成正常数据簇，再手动添加远离簇中心的异常点。
    """
    rng = np.random.RandomState(random_state)
    X, y = make_blobs(
        n_samples=n_samples, n_features=n_features,
        centers=3, cluster_std=1.0, random_state=random_state
    )

    outlier_shift = rng.uniform(15, 25, size=(n_outliers, n_features))
    outlier_sign = rng.choice([-1, 1], size=(n_outliers, n_features))
    X_outliers = X.mean(axis=0) + outlier_shift * outlier_sign

    X_full = np.vstack([X, X_outliers])
    y_full = np.concatenate([y, np.full(n_outliers, -1)])

    return X_full, y_full


def compute_anomaly_scores(model, X):
    """计算重构误差作为异常分数"""
    X_recon = model.reconstruct(X)
    errors = np.mean((X - X_recon) ** 2, axis=1)
    return errors


def evaluate_anomaly_detection(scores, y_full, threshold_percentile=95):
    """评估异常检测效果，返回预测标签和准确率"""
    threshold = np.percentile(scores, threshold_percentile)
    y_pred = (scores > threshold).astype(int)
    y_true = (y_full == -1).astype(int)
    accuracy = np.mean(y_pred == y_true)
    return y_pred, accuracy, threshold


# ============================================================
# 5. 可视化
# ============================================================

def plot_original_vs_reconstructed(X_original, X_reconstructed, y, title="原始数据 vs 重构数据"):
    """绘制原始数据与重构数据的散点图对比（前两个特征）"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    scatter1 = axes[0].scatter(X_original[:, 0], X_original[:, 1], c=y, cmap='viridis', alpha=0.6, s=20)
    axes[0].set_title('原始数据（前两个特征）')
    axes[0].set_xlabel('特征 1')
    axes[0].set_ylabel('特征 2')
    plt.colorbar(scatter1, ax=axes[0])

    scatter2 = axes[1].scatter(X_reconstructed[:, 0], X_reconstructed[:, 1], c=y, cmap='viridis', alpha=0.6, s=20)
    axes[1].set_title('重构数据（前两个特征）')
    axes[1].set_xlabel('特征 1')
    axes[1].set_ylabel('特征 2')
    plt.colorbar(scatter2, ax=axes[1])

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_latent_space(Z, y, title="潜在空间可视化"):
    """绘制2D潜在空间散点图，按类别着色"""
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap='viridis', alpha=0.7, s=25)
    ax.set_title(title)
    ax.set_xlabel('潜在维度 1')
    ax.set_ylabel('潜在维度 2')
    plt.colorbar(scatter, ax=ax, label='类别')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_anomaly_scores(ae_scores, vae_scores, y_full, ae_threshold, vae_threshold):
    """绘制AE和VAE异常分数分布直方图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    normal_ae = ae_scores[y_full != -1]
    outlier_ae = ae_scores[y_full == -1]
    axes[0].hist(normal_ae, bins=30, alpha=0.7, label='正常', color='#3498db', density=True)
    axes[0].hist(outlier_ae, bins=15, alpha=0.7, label='异常', color='#e74c3c', density=True)
    axes[0].axvline(ae_threshold, color='black', linestyle='--', label=f'阈值={ae_threshold:.4f}')
    axes[0].set_title('AE 异常分数分布')
    axes[0].set_xlabel('重构误差')
    axes[0].set_ylabel('密度')
    axes[0].legend()

    normal_vae = vae_scores[y_full != -1]
    outlier_vae = vae_scores[y_full == -1]
    axes[1].hist(normal_vae, bins=30, alpha=0.7, label='正常', color='#3498db', density=True)
    axes[1].hist(outlier_vae, bins=15, alpha=0.7, label='异常', color='#e74c3c', density=True)
    axes[1].axvline(vae_threshold, color='black', linestyle='--', label=f'阈值={vae_threshold:.4f}')
    axes[1].set_title('VAE 异常分数分布')
    axes[1].set_xlabel('重构误差')
    axes[1].set_ylabel('密度')
    axes[1].legend()

    fig.suptitle('异常检测：重构误差分布对比', fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_vae_samples(X_samples, title="VAE 生成样本"):
    """绘制VAE从标准正态先验采样的生成数据散点图（前两个特征）"""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X_samples[:, 0], X_samples[:, 1], alpha=0.5, s=20, color='#2ecc71')
    ax.set_title(title)
    ax.set_xlabel('特征 1')
    ax.set_ylabel('特征 2')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_training_loss(ae_losses, dae_losses, vae_losses, vae_kl_losses=None):
    """绘制各模型训练损失曲线"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(ae_losses, color='#3498db', linewidth=1.5)
    axes[0].set_title('AE 训练损失')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(dae_losses, color='#e67e22', linewidth=1.5)
    axes[1].set_title('DAE 训练损失')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MSE Loss')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(vae_losses, color='#2ecc71', linewidth=1.5, label='Total Loss')
    if vae_kl_losses is not None:
        ax2 = axes[2].twinx()
        ax2.plot(vae_kl_losses, color='#e74c3c', linewidth=1.5, alpha=0.7, label='KL Loss')
        ax2.set_ylabel('KL Loss', color='#e74c3c')
        ax2.tick_params(axis='y', labelcolor='#e74c3c')
        lines1, labels1 = axes[2].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axes[2].legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    axes[2].set_title('VAE 训练损失')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Total Loss')
    axes[2].grid(True, alpha=0.3)

    fig.suptitle('训练损失曲线对比', fontsize=14)
    plt.tight_layout()
    plt.show()


# ============================================================
# 主程序
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("自编码器与VAE实验")
    print("=" * 60)

    # ==== 数据准备 ====
    print("\n--- 数据准备 ---")
    X_full, y_full = create_anomaly_data(
        n_samples=300, n_features=10, n_outliers=15, random_state=42
    )
    n_normal = np.sum(y_full != -1)
    n_outlier = np.sum(y_full == -1)
    print(f"正常样本数: {n_normal}, 异常样本数: {n_outlier}")
    print(f"数据维度: {X_full.shape}")

    X_scaled, X_min, X_range = min_max_scale(X_full)
    print(f"归一化后范围: [{X_scaled.min():.4f}, {X_scaled.max():.4f}]")

    input_dim = X_scaled.shape[1]

    # ==== 1. 自编码器(AE) ====
    print("\n" + "=" * 60)
    print("1. 自编码器(AE)")
    print("=" * 60)
    ae = AutoencoderManual(input_dim=input_dim, hidden_dim=16, latent_dim=2, random_state=42)
    print("训练 AE ...")
    ae.fit(X_scaled, epochs=300, lr=0.001)

    ae_recon = ae.reconstruct(X_scaled)
    ae_recon_original = min_max_inverse(ae_recon, X_min, X_range)
    ae_final_loss = np.mean((X_scaled - ae_recon) ** 2)
    print(f"AE 最终重构误差: {ae_final_loss:.6f}")

    Z_ae = ae.encode(X_scaled)
    print(f"AE 潜在空间范围: [{Z_ae.min():.4f}, {Z_ae.max():.4f}]")

    # ==== 2. 去噪自编码器(DAE) ====
    print("\n" + "=" * 60)
    print("2. 去噪自编码器(DAE)")
    print("=" * 60)
    dae = DenoisingAutoencoder(input_dim=input_dim, hidden_dim=16, latent_dim=2, random_state=42)
    print("训练 DAE (noise_factor=0.3) ...")
    dae.fit(X_scaled, epochs=300, lr=0.001, noise_factor=0.3)

    dae_recon = dae.reconstruct(X_scaled)
    dae_final_loss = np.mean((X_scaled - dae_recon) ** 2)
    print(f"DAE 最终重构误差: {dae_final_loss:.6f}")

    Z_dae = dae.encode(X_scaled)
    print(f"DAE 潜在空间范围: [{Z_dae.min():.4f}, {Z_dae.max():.4f}]")

    # ==== 3. 变分自编码器(VAE) ====
    print("\n" + "=" * 60)
    print("3. 变分自编码器(VAE)")
    print("=" * 60)
    vae = VAEManual(input_dim=input_dim, hidden_dim=16, latent_dim=2, random_state=42)
    print("训练 VAE ...")
    vae.fit(X_scaled, epochs=300, lr=0.001)

    vae_recon = vae.reconstruct(X_scaled)
    vae_recon_original = min_max_inverse(vae_recon, X_min, X_range)
    vae_final_loss = np.mean((X_scaled - vae_recon) ** 2)
    print(f"VAE 最终重构误差: {vae_final_loss:.6f}")

    mu_vae, logvar_vae = vae.encode(X_scaled)
    print(f"VAE 均值范围: [{mu_vae.min():.4f}, {mu_vae.max():.4f}]")
    print(f"VAE 对数方差范围: [{logvar_vae.min():.4f}, {logvar_vae.max():.4f}]")

    # ==== 4. 异常检测 ====
    print("\n" + "=" * 60)
    print("4. 异常检测应用")
    print("=" * 60)
    ae_scores = compute_anomaly_scores(ae, X_scaled)
    vae_scores = compute_anomaly_scores(vae, X_scaled)

    ae_pred, ae_acc, ae_thresh = evaluate_anomaly_detection(ae_scores, y_full, threshold_percentile=95)
    vae_pred, vae_acc, vae_thresh = evaluate_anomaly_detection(vae_scores, y_full, threshold_percentile=95)

    print(f"AE  异常检测准确率: {ae_acc:.4f} (阈值: {ae_thresh:.6f})")
    print(f"VAE 异常检测准确率: {vae_acc:.4f} (阈值: {vae_thresh:.6f})")

    ae_outlier_scores = ae_scores[y_full == -1]
    ae_normal_scores = ae_scores[y_full != -1]
    vae_outlier_scores = vae_scores[y_full == -1]
    vae_normal_scores = vae_scores[y_full != -1]

    print(f"\nAE  正常样本平均重构误差: {ae_normal_scores.mean():.6f}")
    print(f"AE  异常样本平均重构误差: {ae_outlier_scores.mean():.6f}")
    print(f"VAE 正常样本平均重构误差: {vae_normal_scores.mean():.6f}")
    print(f"VAE 异常样本平均重构误差: {vae_outlier_scores.mean():.6f}")

    # ==== 5. VAE 生成样本 ====
    print("\n" + "=" * 60)
    print("5. VAE 生成样本")
    print("=" * 60)
    vae_samples = vae.sample(n_samples=200)
    vae_samples_original = min_max_inverse(vae_samples, X_min, X_range)
    print(f"生成样本数: {vae_samples.shape[0]}")
    print(f"生成样本范围: [{vae_samples_original.min():.4f}, {vae_samples_original.max():.4f}]")

    # ==== 6. 可视化 ====
    print("\n" + "=" * 60)
    print("6. 可视化")
    print("=" * 60)

    plot_training_loss(ae.loss_history, dae.loss_history, vae.loss_history, vae.kl_history)

    plot_original_vs_reconstructed(
        X_full, ae_recon_original, y_full, title="AE: 原始数据 vs 重构数据"
    )

    plot_original_vs_reconstructed(
        X_full, vae_recon_original, y_full, title="VAE: 原始数据 vs 重构数据"
    )

    plot_latent_space(Z_ae, y_full, title="AE 潜在空间可视化")
    plot_latent_space(Z_dae, y_full, title="DAE 潜在空间可视化")
    plot_latent_space(mu_vae, y_full, title="VAE 潜在空间可视化（均值）")

    plot_anomaly_scores(ae_scores, vae_scores, y_full, ae_thresh, vae_thresh)

    plot_vae_samples(vae_samples_original, title="VAE 生成样本（从标准正态先验采样）")

    print("\n实验完成！")
