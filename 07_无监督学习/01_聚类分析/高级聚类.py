"""
高级聚类模块 - Advanced Clustering
===================================
涵盖K-Means之外的高级聚类方法：
1. DBSCAN密度聚类
2. 层次聚类 (Agglomerative)
3. 高斯混合模型 (GMM)
4. 谱聚类 (Spectral Clustering)
5. 聚类方法对比
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.cluster import DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from utils import setup_chinese_font


# ============================================================
# 1. DBSCAN密度聚类
# ============================================================
def demo_dbscan():
    """DBSCAN密度聚类 — 可发现任意形状簇，自动识别噪声点"""
    print("--- DBSCAN密度聚类 ---")
    X, y_true = make_moons(n_samples=500, noise=0.1, random_state=42)

    db = DBSCAN(eps=0.2, min_samples=5)
    labels = db.fit_predict(X)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"  发现簇数: {n_clusters}, 噪声点: {n_noise}")
    if n_clusters > 1:
        sil = silhouette_score(X[labels != -1], labels[labels != -1])
        print(f"  轮廓系数: {sil:.4f}")

    _plot_clusters(X, labels, 'DBSCAN聚类 (月牙形数据)')


# ============================================================
# 2. 层次聚类
# ============================================================
def demo_agglomerative():
    """层次聚类 — Agglomerative自底向上"""
    print("--- 层次聚类 ---")
    X, y_true = make_blobs(n_samples=200, centers=3, random_state=42)

    # 绘制树状图
    Z = linkage(X, method='ward')
    plt.figure(figsize=(10, 5))
    dendrogram(Z, truncate_mode='lastp', p=20, leaf_font_size=10)
    plt.title('层次聚类树状图 (Ward链接)')
    plt.xlabel('样本索引')
    plt.ylabel('距离')
    plt.savefig('层次聚类树状图.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 聚类
    agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
    labels = agg.fit_predict(X)
    print(f"  轮廓系数: {silhouette_score(X, labels):.4f}")

    _plot_clusters(X, labels, '层次聚类 (Ward链接)')


# ============================================================
# 3. 高斯混合模型 (GMM)
# ============================================================
def demo_gmm():
    """高斯混合模型 — 软聚类，给出概率"""
    print("--- 高斯混合模型 (GMM) ---")
    X, y_true = make_blobs(n_samples=400, centers=3, cluster_std=1.0, random_state=42)

    gmm = GaussianMixture(n_components=3, random_state=42)
    labels = gmm.fit_predict(X)
    probs = gmm.predict_proba(X)

    print(f"  轮廓系数: {silhouette_score(X, labels):.4f}")
    print(f"  BIC: {gmm.bic(X):.2f}, AIC: {gmm.aic(X):.2f}")

    # BIC选择最优成分数
    bics = []
    aics = []
    k_range = range(2, 8)
    for k in k_range:
        gm = GaussianMixture(n_components=k, random_state=42)
        gm.fit(X)
        bics.append(gm.bic(X))
        aics.append(gm.aic(X))

    plt.figure(figsize=(8, 4))
    plt.plot(k_range, bics, 'bo-', label='BIC')
    plt.plot(k_range, aics, 'ro-', label='AIC')
    plt.xlabel('成分数')
    plt.ylabel('信息准则')
    plt.title('GMM模型选择 (BIC/AIC)')
    plt.legend()
    plt.savefig('GMM模型选择.png', dpi=150, bbox_inches='tight')
    plt.show()

    _plot_clusters(X, labels, 'GMM聚类结果')


# ============================================================
# 4. 谱聚类
# ============================================================
def demo_spectral():
    """谱聚类 — 适用于非凸形状数据"""
    print("--- 谱聚类 ---")
    X, y_true = make_circles(n_samples=500, factor=0.5, noise=0.05, random_state=42)

    sc = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=42)
    labels = sc.fit_predict(X)
    print(f"  轮廓系数: {silhouette_score(X, labels):.4f}")

    _plot_clusters(X, labels, '谱聚类 (同心圆数据)')


# ============================================================
# 5. 聚类方法对比
# ============================================================
def compare_clustering_methods():
    """不同聚类方法在多种数据分布上的对比"""
    print("--- 聚类方法对比 ---")
    datasets = {
        '月牙形': make_moons(n_samples=500, noise=0.1, random_state=42)[0],
        '同心圆': make_circles(n_samples=500, factor=0.5, noise=0.05, random_state=42)[0],
        '高斯簇': make_blobs(n_samples=500, centers=3, random_state=42)[0],
    }

    methods = {
        'K-Means': lambda: KMeans_sk(n_clusters=2),
        'DBSCAN': lambda: DBSCAN(eps=0.3, min_samples=5),
        '层次聚类': lambda: AgglomerativeClustering(n_clusters=2),
        'GMM': lambda: GaussianMixture(n_components=2, random_state=42),
        '谱聚类': lambda: SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=42),
    }

    # 延迟导入避免冲突
    from sklearn.cluster import KMeans as KMeans_sk

    fig, axes = plt.subplots(len(datasets), len(methods), figsize=(20, 10))

    for i, (data_name, X) in enumerate(datasets.items()):
        for j, (method_name, method_fn) in enumerate(methods.items()):
            model = method_fn()
            if method_name == 'GMM':
                labels = model.fit_predict(X)
            else:
                labels = model.fit_predict(X)

            ax = axes[i][j]
            ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='Set1', s=10, alpha=0.6)

            if i == 0:
                ax.set_title(method_name)
            if j == 0:
                ax.set_ylabel(data_name)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle('聚类方法对比', fontsize=14)
    plt.tight_layout()
    plt.savefig('聚类方法对比.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 6. DBSCAN 手动实现
# ============================================================
class DBSCANManual:
    """DBSCAN（基于密度的空间聚类应用）手动实现

    核心概念:
    - ε-邻域: 以某点为圆心、eps 为半径的区域
    - 核心点: ε-邻域内至少有 min_samples 个点的点
    - 边界点: 不是核心点，但在某个核心点的 ε-邻域内
    - 噪声点: 既不是核心点也不是边界点（标记为 -1）

    算法流程:
    1. 计算所有点之间的距离矩阵
    2. 对每个点，找出其 ε-邻域内的所有邻居
    3. 识别核心点（邻居数 >= min_samples）
    4. 从未访问的核心点出发，深度优先扩展簇
    5. 核心点的邻居若也是核心点，则继续扩展；否则标记为边界点
    6. 未被任何簇包含的点标记为噪声（-1）
    """

    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def fit(self, X):
        """执行 DBSCAN 聚类

        参数:
            X: 输入数据, shape=(n_samples, n_features)

        返回:
            self
        """
        n_samples = X.shape[0]
        self.labels_ = np.full(n_samples, -1, dtype=int)

        # 步骤1: 计算距离矩阵
        # 利用广播机制: diff[i,j] = ||X[i] - X[j]||^2
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))

        # 步骤2: 找出每个点的 ε-邻域邻居
        neighbors = []
        for i in range(n_samples):
            # 距离 <= eps 的所有点（包括自身）
            neighbor_indices = np.where(dist_matrix[i] <= self.eps)[0]
            neighbors.append(neighbor_indices)

        # 步骤3: 识别核心点
        # 核心点: ε-邻域内点数 >= min_samples
        is_core = np.array([len(nbrs) >= self.min_samples for nbrs in neighbors])

        # 步骤4: 从核心点出发，深度优先扩展簇
        cluster_id = 0
        visited = np.zeros(n_samples, dtype=bool)

        for i in range(n_samples):
            # 跳过已访问的点或非核心点
            if visited[i] or not is_core[i]:
                continue

            # 开始新簇
            self.labels_[i] = cluster_id
            visited[i] = True

            # 使用队列进行广度优先扩展
            queue = list(neighbors[i])
            while queue:
                j = queue.pop(0)

                if visited[j]:
                    continue
                visited[j] = True

                # 如果 j 是核心点，将其邻居加入队列继续扩展
                if is_core[j]:
                    self.labels_[j] = cluster_id
                    for nbr in neighbors[j]:
                        if not visited[nbr]:
                            queue.append(nbr)
                else:
                    # j 是边界点，归入当前簇但不扩展
                    if self.labels_[j] == -1:
                        self.labels_[j] = cluster_id

            cluster_id += 1

        return self

    def predict(self, X):
        """返回聚类标签（-1 表示噪声）

        参数:
            X: 输入数据（需与 fit 时相同的数据）

        返回:
            labels: 聚类标签数组, shape=(n_samples,)
        """
        if self.labels_ is None:
            raise RuntimeError("请先调用 fit 方法")
        return self.labels_


# ============================================================
# 辅助函数
# ============================================================
def _plot_clusters(X, labels, title):
    plt.figure(figsize=(8, 6))
    unique_labels = set(labels)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        mask = labels == k
        if k == -1:
            plt.scatter(X[mask, 0], X[mask, 1], c='gray', s=10, alpha=0.3, label='噪声')
        else:
            plt.scatter(X[mask, 0], X[mask, 1], c=[col], s=20, alpha=0.6, label=f'簇{k}')
    plt.title(title)
    plt.legend()
    plt.savefig(f'{title}.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 主程序演示
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("高级聚类方法完整流程演示")
    print("=" * 60)

    demo_dbscan()
    demo_agglomerative()
    demo_gmm()
    demo_spectral()
    compare_clustering_methods()

    # ============================================================
    # DBSCAN 手动实现 vs sklearn 对比
    # ============================================================
    print("\n" + "=" * 60)
    print("DBSCAN 手动实现 vs sklearn 对比")
    print("=" * 60)

    X_moons, y_true = make_moons(n_samples=500, noise=0.1, random_state=42)

    # 手动实现
    db_manual = DBSCANManual(eps=0.2, min_samples=5)
    db_manual.fit(X_moons)
    labels_manual = db_manual.predict(X_moons)
    n_clusters_manual = len(set(labels_manual)) - (1 if -1 in labels_manual else 0)
    n_noise_manual = (labels_manual == -1).sum()
    print(f"  DBSCANManual: 簇数={n_clusters_manual}, 噪声点={n_noise_manual}")
    if n_clusters_manual > 1:
        sil_manual = silhouette_score(X_moons[labels_manual != -1], labels_manual[labels_manual != -1])
        print(f"  DBSCANManual 轮廓系数: {sil_manual:.4f}")

    # sklearn 实现
    db_sklearn = DBSCAN(eps=0.2, min_samples=5)
    labels_sklearn = db_sklearn.fit_predict(X_moons)
    n_clusters_sklearn = len(set(labels_sklearn)) - (1 if -1 in labels_sklearn else 0)
    n_noise_sklearn = (labels_sklearn == -1).sum()
    print(f"  sklearn DBSCAN: 簇数={n_clusters_sklearn}, 噪声点={n_noise_sklearn}")
    if n_clusters_sklearn > 1:
        sil_sklearn = silhouette_score(X_moons[labels_sklearn != -1], labels_sklearn[labels_sklearn != -1])
        print(f"  sklearn DBSCAN 轮廓系数: {sil_sklearn:.4f}")

    # 可视化对比
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, labels, title in [
        (axes[0], labels_manual, 'DBSCANManual 手动实现'),
        (axes[1], labels_sklearn, 'sklearn DBSCAN')
    ]:
        unique_labels = set(labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        for k, col in zip(unique_labels, colors):
            mask = labels == k
            if k == -1:
                ax.scatter(X_moons[mask, 0], X_moons[mask, 1], c='gray', s=10, alpha=0.3, label='噪声')
            else:
                ax.scatter(X_moons[mask, 0], X_moons[mask, 1], c=[col], s=20, alpha=0.6, label=f'簇{k}')
        ax.set_title(title)
        ax.legend()

    plt.suptitle('DBSCAN 手动实现 vs sklearn 对比', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'DBSCAN对比.png'), dpi=150, bbox_inches='tight')
    plt.show()
