"""对比学习与自监督学习
=====================
本模块系统介绍对比学习与自监督学习的核心概念与实现：
1. 自监督学习概述：预训练任务设计、对比学习范式、主流方法
2. 数据增强函数：高斯噪声、随机掩码、随机裁剪、颜色抖动
3. 对比损失函数：NT-Xent损失的手动实现
4. SimCLR简化实现：基于numpy的完整对比学习框架
5. 线性评估协议：对比学习特征质量评估
6. 可视化：t-SNE特征可视化、训练损失曲线、准确率对比
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import setup_chinese_font

setup_chinese_font()


# ====自监督学习概述====

def print_overview():
    print("=" * 60)
    print("自监督学习概述")
    print("=" * 60)
    print("\n【预训练任务设计】")
    print("  1. 掩码预测 (Masked Prediction):")
    print("     随机遮蔽输入部分内容，模型预测被遮蔽的部分")
    print("     代表: BERT (掩码语言模型), MAE (掩码自编码器)")
    print("  2. 旋转预测 (Rotation Prediction):")
    print("     对输入施加旋转变换，模型预测旋转角度")
    print("     代表: RotNet (Gidaris et al., 2018)")
    print("  3. 拼图重组 (Jigsaw Puzzle):")
    print("     将输入切分为多个块并打乱顺序，模型预测原始排列")
    print("     代表: Noroozi & Favaro, 2016")
    print("\n【对比学习范式】")
    print("  核心思想: 拉近正样本对，推远负样本对")
    print("  正样本对: 同一样本的不同增强视图")
    print("  负样本对: 不同样本的增强视图")
    print("  目标: 学习对数据增强不变的特征表示")
    print("\n【主流方法】")
    print("  1. SimCLR (Chen et al., 2020):")
    print("     大批量对比学习 + 非线性投影头 + 强数据增强")
    print("  2. MoCo (He et al., 2020):")
    print("     动量编码器 + 队列机制，解决负样本数量限制")
    print("  3. BYOL (Grill et al., 2020):")
    print("     无需负样本，在线网络与目标网络互相学习")
    print("  4. SwAV (Caron et al., 2020):")
    print("     在线聚类分配，结合对比学习与聚类优势")


# ====数据增强函数====

def gaussian_noise(X, scale=0.1):
    noise = np.random.normal(0, scale, size=X.shape)
    return X + noise


def random_mask(X, mask_ratio=0.2):
    mask = np.random.binomial(1, 1 - mask_ratio, size=X.shape)
    return X * mask


def random_crop(X, crop_ratio=0.8):
    n_samples, n_features = X.shape
    crop_size = max(1, int(n_features * crop_ratio))
    X_crop = np.zeros_like(X)
    max_start = max(1, n_features - crop_size + 1)
    for i in range(n_samples):
        start = np.random.randint(0, max_start)
        cropped = X[i, start:start + crop_size]
        original_indices = np.linspace(0, crop_size - 1, n_features)
        crop_indices = np.arange(crop_size)
        X_crop[i] = np.interp(original_indices, crop_indices, cropped)
    return X_crop


def color_jitter_1d(X, brightness=0.1, contrast=0.1):
    X_jitter = X.copy()
    brightness_shift = np.random.uniform(-brightness, brightness, size=X.shape)
    X_jitter += brightness_shift
    contrast_scale = np.random.uniform(1 - contrast, 1 + contrast, size=X.shape)
    X_jitter *= contrast_scale
    return X_jitter


# ====对比损失函数====

def nt_xent_loss(features1, features2, temperature=0.5):
    norms1 = np.linalg.norm(features1, axis=1, keepdims=True)
    norms2 = np.linalg.norm(features2, axis=1, keepdims=True)
    z1 = features1 / (norms1 + 1e-8)
    z2 = features2 / (norms2 + 1e-8)
    sim = z1 @ z2.T / temperature
    sim_max = np.max(sim, axis=1, keepdims=True)
    exp_sim = np.exp(sim - sim_max)
    sum_exp = np.sum(exp_sim, axis=1, keepdims=True)
    log_prob = sim - sim_max - np.log(sum_exp + 1e-8)
    n = features1.shape[0]
    labels = np.arange(n)
    loss = -np.mean(log_prob[labels, labels])
    return loss


# ====SimCLR简化实现====

class SimCLRManual:

    def __init__(self, input_dim, hidden_dim, projection_dim, lr=0.01):
        np.random.seed(42)
        scale_enc = np.sqrt(2.0 / input_dim)
        self.W_enc = np.random.randn(input_dim, hidden_dim) * scale_enc
        self.b_enc = np.zeros(hidden_dim)
        scale_proj = np.sqrt(2.0 / hidden_dim)
        self.W_proj = np.random.randn(hidden_dim, projection_dim) * scale_proj
        self.b_proj = np.zeros(projection_dim)
        self.lr = lr
        self.loss_history = []

    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_grad(self, x):
        return (x > 0).astype(float)

    def _forward(self, X):
        h_pre = X @ self.W_enc + self.b_enc
        h = self._relu(h_pre)
        z_pre = h @ self.W_proj + self.b_proj
        z = self._relu(z_pre)
        norms = np.linalg.norm(z, axis=1, keepdims=True) + 1e-8
        z_norm = z / norms
        return h_pre, h, z_pre, z, z_norm

    def _norm_grad(self, z, z_norm, d_z_norm):
        norms = np.linalg.norm(z, axis=1, keepdims=True) + 1e-8
        proj = np.sum(d_z_norm * z_norm, axis=1, keepdims=True)
        d_z = (d_z_norm - z_norm * proj) / norms
        return d_z

    def fit(self, X, epochs=50, temperature=0.5):
        n = X.shape[0]
        for epoch in range(epochs):
            X1 = gaussian_noise(X)
            X2 = random_mask(X)

            h1_pre, h1, z1_pre, z1, z1_norm = self._forward(X1)
            h2_pre, h2, z2_pre, z2, z2_norm = self._forward(X2)

            sim = z1_norm @ z2_norm.T / temperature
            sim_max = np.max(sim, axis=1, keepdims=True)
            exp_sim = np.exp(sim - sim_max)
            sum_exp = np.sum(exp_sim, axis=1, keepdims=True)
            prob = exp_sim / (sum_exp + 1e-8)

            labels = np.arange(n)
            loss = -np.mean(np.log(prob[labels, labels] + 1e-8))
            self.loss_history.append(loss)

            d_sim = prob.copy()
            d_sim[labels, labels] -= 1
            d_sim /= (n * temperature)

            d_z1_norm = d_sim @ z2_norm
            d_z2_norm = d_sim.T @ z1_norm

            d_z1 = self._norm_grad(z1, z1_norm, d_z1_norm)
            d_z2 = self._norm_grad(z2, z2_norm, d_z2_norm)

            d_z1_pre = d_z1 * self._relu_grad(z1_pre)
            d_z2_pre = d_z2 * self._relu_grad(z2_pre)

            d_W_proj = h1.T @ d_z1_pre + h2.T @ d_z2_pre
            d_b_proj = np.sum(d_z1_pre, axis=0) + np.sum(d_z2_pre, axis=0)

            d_h1 = d_z1_pre @ self.W_proj.T
            d_h2 = d_z2_pre @ self.W_proj.T

            d_h1_pre = d_h1 * self._relu_grad(h1_pre)
            d_h2_pre = d_h2 * self._relu_grad(h2_pre)

            d_W_enc = X1.T @ d_h1_pre + X2.T @ d_h2_pre
            d_b_enc = np.sum(d_h1_pre, axis=0) + np.sum(d_h2_pre, axis=0)

            grad_norm = np.sqrt(
                np.sum(d_W_enc ** 2) + np.sum(d_b_enc ** 2) +
                np.sum(d_W_proj ** 2) + np.sum(d_b_proj ** 2)
            )
            clip_val = 5.0
            if grad_norm > clip_val:
                scale = clip_val / grad_norm
                d_W_enc *= scale
                d_b_enc *= scale
                d_W_proj *= scale
                d_b_proj *= scale

            self.W_proj -= self.lr * d_W_proj
            self.b_proj -= self.lr * d_b_proj
            self.W_enc -= self.lr * d_W_enc
            self.b_enc -= self.lr * d_b_enc

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

    def encode(self, X):
        h = self._relu(X @ self.W_enc + self.b_enc)
        return h


# ====监督学习基线====

class SupervisedNet:

    def __init__(self, input_dim, hidden_dim, n_classes, lr=0.01):
        np.random.seed(42)
        scale1 = np.sqrt(2.0 / input_dim)
        self.W1 = np.random.randn(input_dim, hidden_dim) * scale1
        self.b1 = np.zeros(hidden_dim)
        scale2 = np.sqrt(1.0 / hidden_dim)
        self.W2 = np.random.randn(hidden_dim, n_classes) * scale2
        self.b2 = np.zeros(n_classes)
        self.lr = lr
        self.loss_history = []

    def _relu(self, x):
        return np.maximum(0, x)

    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def fit(self, X, y, epochs=50):
        n = X.shape[0]
        n_classes = self.W2.shape[1]
        one_hot = np.zeros((n, n_classes))
        one_hot[np.arange(n), y] = 1
        for epoch in range(epochs):
            h = self._relu(X @ self.W1 + self.b1)
            logits = h @ self.W2 + self.b2
            prob = self._softmax(logits)

            loss = -np.mean(np.sum(one_hot * np.log(prob + 1e-8), axis=1))
            self.loss_history.append(loss)

            d_logits = (prob - one_hot) / n
            d_W2 = h.T @ d_logits
            d_b2 = np.sum(d_logits, axis=0)
            d_h = d_logits @ self.W2.T
            d_h_pre = d_h * (h > 0).astype(float)
            d_W1 = X.T @ d_h_pre
            d_b1 = np.sum(d_h_pre, axis=0)

            self.W2 -= self.lr * d_W2
            self.b2 -= self.lr * d_b2
            self.W1 -= self.lr * d_W1
            self.b1 -= self.lr * d_b1

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

    def encode(self, X):
        return self._relu(X @ self.W1 + self.b1)

    def predict(self, X):
        h = self._relu(X @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        return np.argmax(logits, axis=1)


# ====线性评估协议====

def linear_evaluation(X_train, y_train, X_test, y_test, encoder=None):
    if encoder is not None:
        features_train = encoder(X_train)
        features_test = encoder(X_test)
    else:
        rng = np.random.RandomState(42)
        random_w = rng.randn(X_train.shape[1], 64) * 0.1
        features_train = X_train @ random_w
        features_test = X_test @ random_w

    scaler = StandardScaler()
    features_train = scaler.fit_transform(features_train)
    features_test = scaler.transform(features_test)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(features_train, y_train)
    y_pred = clf.predict(features_test)
    acc = accuracy_score(y_test, y_pred)
    return acc


# ====可视化====

def visualize_tsne(X, y, encoders, titles, figsize=(18, 5)):
    fig, axes = plt.subplots(1, len(encoders), figsize=figsize)
    for idx, (encoder, title) in enumerate(zip(encoders, titles)):
        if encoder is not None:
            features = encoder(X)
        else:
            features = X
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embedded = tsne.fit_transform(features_scaled)
        ax = axes[idx]
        scatter = ax.scatter(embedded[:, 0], embedded[:, 1], c=y,
                             cmap='tab10', s=10, alpha=0.7)
        ax.set_title(title, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.colorbar(scatter, ax=axes.tolist(), shrink=0.6)
    plt.suptitle('t-SNE 特征可视化对比', fontsize=16)
    plt.tight_layout()
    plt.savefig('tsne_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()


def visualize_loss_curve(loss_histories, labels, title='训练损失曲线'):
    fig, ax = plt.subplots(figsize=(8, 5))
    for loss_hist, label in zip(loss_histories, labels):
        ax.plot(loss_hist, label=label, linewidth=1.5)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('loss_curve.png', dpi=150, bbox_inches='tight')
    plt.show()


def visualize_accuracy_comparison(methods, accuracies):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    bars = ax.bar(methods, accuracies, color=colors, width=0.5,
                  edgecolor='black', linewidth=0.8)
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{acc:.2%}', ha='center', va='bottom',
                fontsize=12, fontweight='bold')
    ax.set_ylabel('准确率', fontsize=12)
    ax.set_title('线性评估准确率对比', fontsize=14)
    ax.set_ylim(0, max(accuracies) * 1.15)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('accuracy_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


# ====主程序====

if __name__ == '__main__':
    np.random.seed(42)

    print_overview()

    print("\n" + "=" * 60)
    print("数据增强函数演示")
    print("=" * 60)

    X_demo = np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                       [0.5, 1.5, 2.5, 3.5, 4.5]])
    print(f"\n原始数据:\n{X_demo}")
    print(f"\n高斯噪声增强:\n{gaussian_noise(X_demo)}")
    print(f"\n随机掩码增强:\n{random_mask(X_demo)}")
    print(f"\n随机裁剪增强:\n{random_crop(X_demo)}")
    print(f"\n颜色抖动增强:\n{color_jitter_1d(X_demo)}")

    print("\n" + "=" * 60)
    print("NT-Xent 对比损失演示")
    print("=" * 60)

    np.random.seed(42)
    feat1 = np.random.randn(4, 8)
    feat2 = feat1 + np.random.randn(4, 8) * 0.1
    loss_val = nt_xent_loss(feat1, feat2, temperature=0.5)
    print(f"\n正样本对(相似特征) NT-Xent Loss: {loss_val:.4f}")

    feat3 = np.random.randn(4, 8)
    loss_val2 = nt_xent_loss(feat1, feat3, temperature=0.5)
    print(f"负样本对(随机特征) NT-Xent Loss: {loss_val2:.4f}")

    print("\n" + "=" * 60)
    print("SimCLR 对比学习训练")
    print("=" * 60)

    X, y = make_blobs(n_samples=500, n_features=20, centers=4,
                       cluster_std=1.5, random_state=42)
    scaler_data = StandardScaler()
    X = scaler_data.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    simclr = SimCLRManual(input_dim=20, hidden_dim=64,
                           projection_dim=32, lr=0.01)
    print("\n训练 SimCLR...")
    simclr.fit(X_train, epochs=100, temperature=0.5)

    print("\n" + "=" * 60)
    print("监督学习基线训练")
    print("=" * 60)

    supervised = SupervisedNet(input_dim=20, hidden_dim=64,
                               n_classes=4, lr=0.01)
    print("\n训练监督网络...")
    supervised.fit(X_train, y_train, epochs=100)

    print("\n" + "=" * 60)
    print("线性评估协议")
    print("=" * 60)

    acc_random = linear_evaluation(X_train, y_train, X_test, y_test,
                                    encoder=None)
    acc_simclr = linear_evaluation(X_train, y_train, X_test, y_test,
                                    encoder=simclr.encode)
    acc_supervised = linear_evaluation(X_train, y_train, X_test, y_test,
                                        encoder=supervised.encode)

    print(f"\n随机特征 + 线性分类器 准确率: {acc_random:.2%}")
    print(f"SimCLR特征 + 线性分类器 准确率: {acc_simclr:.2%}")
    print(f"监督学习特征 + 线性分类器 准确率: {acc_supervised:.2%}")

    print("\n" + "=" * 60)
    print("可视化")
    print("=" * 60)

    def random_encode(X_data):
        rng = np.random.RandomState(42)
        random_w = rng.randn(X_data.shape[1], 64) * 0.1
        return X_data @ random_w

    visualize_tsne(
        X_test, y_test,
        encoders=[random_encode, simclr.encode, supervised.encode],
        titles=['随机特征', 'SimCLR特征', '监督学习特征']
    )

    visualize_loss_curve(
        [simclr.loss_history, supervised.loss_history],
        ['SimCLR (NT-Xent)', 'Supervised (Cross-Entropy)']
    )

    visualize_accuracy_comparison(
        ['随机特征\n+线性分类器', 'SimCLR特征\n+线性分类器',
         '监督学习特征\n+线性分类器'],
        [acc_random, acc_simclr, acc_supervised]
    )

    print("\n对比学习与自监督学习演示完成！")
