"""
K-Means聚类模块 - K-Means Clustering
======================================
涵盖K-Means聚类的核心内容：
1. K-Means算法手动实现
2. K值选择 (肘部法则、轮廓系数)
3. Mini-Batch K-Means
4. 聚类可视化
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, load_iris
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from utils import setup_chinese_font

setup_chinese_font()


# ============================================================
# 1. K-Means手动实现
# ============================================================
class KMeansManual:
    """K-Means聚类 — 手动实现"""

    def __init__(self, n_clusters=3, max_iters=100, random_state=42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state

    def fit(self, X):
        np.random.seed(self.random_state)
        m, n = X.shape

        # 随机初始化中心点
        idx = np.random.choice(m, self.n_clusters, replace=False)
        self.centroids = X[idx].copy()

        for iteration in range(self.max_iters):
            # 分配每个点到最近中心
            distances = np.array([np.linalg.norm(X - c, axis=1) for c in self.centroids]).T
            self.labels = np.argmin(distances, axis=1)

            # 更新中心
            new_centroids = np.array([X[self.labels == k].mean(axis=0) for k in range(self.n_clusters)])

            # 检查收敛
            if np.allclose(self.centroids, new_centroids):
                print(f"  K-Means在第{iteration + 1}轮收敛")
                break
            self.centroids = new_centroids

        # 计算WCSS（簇内平方和）
        self.inertia_ = sum(
            np.sum((X[self.labels == k] - self.centroids[k]) ** 2)
            for k in range(self.n_clusters)
        )
        return self

    def predict(self, X):
        distances = np.array([np.linalg.norm(X - c, axis=1) for c in self.centroids]).T
        return np.argmin(distances, axis=1)


# ============================================================
# 2. K值选择
# ============================================================
def find_optimal_k(X, k_range=range(2, 11)):
    """肘部法则 + 轮廓系数选择最优K值"""
    inertias = []
    silhouettes = []
    ch_scores = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))
        ch_scores.append(calinski_harabasz_score(X, labels))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(k_range, inertias, 'bo-')
    axes[0].set_xlabel('K值')
    axes[0].set_ylabel('WCSS (簇内平方和)')
    axes[0].set_title('肘部法则')

    axes[1].plot(k_range, silhouettes, 'ro-')
    axes[1].set_xlabel('K值')
    axes[1].set_ylabel('轮廓系数')
    axes[1].set_title('轮廓系数法')

    axes[2].plot(k_range, ch_scores, 'go-')
    axes[2].set_xlabel('K值')
    axes[2].set_ylabel('CH指数')
    axes[2].set_title('Calinski-Harabasz指数')

    plt.tight_layout()
    plt.savefig('K值选择.png', dpi=150, bbox_inches='tight')
    plt.show()

    best_k_sil = list(k_range)[np.argmax(silhouettes)]
    print(f"轮廓系数最优K值: {best_k_sil}")
    return best_k_sil


# ============================================================
# 3. 聚类可视化
# ============================================================
def plot_clusters(X, labels, centroids, title='K-Means聚类结果'):
    """可视化聚类结果"""
    plt.figure(figsize=(8, 6))
    n_clusters = len(np.unique(labels))
    colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))

    for k in range(n_clusters):
        mask = labels == k
        plt.scatter(X[mask, 0], X[mask, 1], c=[colors[k]], s=30, alpha=0.6, label=f'簇{k}')

    if centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=200, label='中心点')

    plt.title(title)
    plt.legend()
    plt.savefig('KMeans聚类结果.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 主程序演示
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("K-Means聚类完整流程演示")
    print("=" * 60)

    # --- 1. 手动实现K-Means ---
    print("\n--- 1. K-Means手动实现 ---")
    X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

    model = KMeansManual(n_clusters=4)
    model.fit(X)
    print(f"  WCSS: {model.inertia_:.2f}")
    print(f"  轮廓系数: {silhouette_score(X, model.labels):.4f}")
    plot_clusters(X, model.labels, model.centroids, '手动实现K-Means')

    # --- 2. K值选择 ---
    print("\n--- 2. K值选择 ---")
    best_k = find_optimal_k(X)

    # --- 3. sklearn K-Means ---
    print("\n--- 3. sklearn K-Means (Iris数据集) ---")
    iris = load_iris()
    X_iris = StandardScaler().fit_transform(iris.data)

    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = km.fit_predict(X_iris)
    print(f"  轮廓系数: {silhouette_score(X_iris, labels):.4f}")
    print(f"  CH指数: {calinski_harabasz_score(X_iris, labels):.2f}")

    # --- 4. Mini-Batch K-Means ---
    print("\n--- 4. Mini-Batch K-Means ---")
    X_large, _ = make_blobs(n_samples=5000, centers=5, random_state=42)

    mbkm = MiniBatchKMeans(n_clusters=5, batch_size=100, random_state=42)
    mbkm.fit(X_large)
    print(f"  Mini-Batch K-Means WCSS: {mbkm.inertia_:.2f}")

    km_full = KMeans(n_clusters=5, random_state=42, n_init=10)
    km_full.fit(X_large)
    print(f"  标准K-Means WCSS:       {km_full.inertia_:.2f}")
