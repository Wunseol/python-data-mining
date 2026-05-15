"""
异常检测模块 - Anomaly Detection
=================================
涵盖常见的异常检测方法：
1. 统计方法 (Z-Score、IQR)
2. 孤立森林 (Isolation Forest)
3. LOF局部异常因子
4. One-Class SVM
5. 基于重构误差的异常检测 (PCA)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, load_wine
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 辅助函数：创建含异常值的示例数据
# ============================================================
def create_anomaly_data(n_normal=300, n_outliers=20, random_state=42):
    """创建含异常值的二维数据"""
    np.random.seed(random_state)
    X_normal = np.random.randn(n_normal, 2) * 0.5
    X_outliers = np.random.uniform(low=-4, high=4, size=(n_outliers, 2))
    X = np.vstack([X_normal, X_outliers])
    y = np.hstack([np.zeros(n_normal), np.ones(n_outliers)])  # 0=正常, 1=异常
    return X, y


def plot_anomalies(X, labels, title, normal_label=0):
    """可视化异常检测结果"""
    plt.figure(figsize=(8, 6))
    mask_normal = labels == normal_label
    mask_anomaly = labels != normal_label

    plt.scatter(X[mask_normal, 0], X[mask_normal, 1], c='blue', s=20,
                alpha=0.5, label='正常点')
    plt.scatter(X[mask_anomaly, 0], X[mask_anomaly, 1], c='red', s=40,
                alpha=0.8, label='异常点', marker='x')
    plt.title(title)
    plt.legend()
    plt.savefig(f'{title}.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 1. 统计方法 — Z-Score
# ============================================================
def detect_zscore(X, threshold=3):
    """Z-Score异常检测：超过阈值的为异常"""
    z_scores = np.abs((X - X.mean(axis=0)) / X.std(axis=0))
    is_anomaly = np.any(z_scores > threshold, axis=1)
    labels = np.where(is_anomaly, 1, 0)
    print(f"Z-Score检测: 阈值={threshold}, 异常数={is_anomaly.sum()}/{len(X)}")
    return labels


# ============================================================
# 2. 统计方法 — IQR
# ============================================================
def detect_iqr(X, factor=1.5):
    """IQR异常检测"""
    Q1 = np.percentile(X, 25, axis=0)
    Q3 = np.percentile(X, 75, axis=0)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    is_anomaly = np.any((X < lower) | (X > upper), axis=1)
    labels = np.where(is_anomaly, 1, 0)
    print(f"IQR检测: 因子={factor}, 异常数={is_anomaly.sum()}/{len(X)}")
    return labels


# ============================================================
# 3. 孤立森林 (Isolation Forest)
# ============================================================
def detect_isolation_forest(X, contamination=0.05, random_state=42):
    """孤立森林异常检测"""
    iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
    pred = iso_forest.fit_predict(X)
    labels = np.where(pred == -1, 1, 0)
    scores = iso_forest.decision_function(X)
    print(f"孤立森林检测: contamination={contamination}, 异常数={labels.sum()}/{len(X)}")
    return labels, scores, iso_forest


# ============================================================
# 4. LOF局部异常因子
# ============================================================
def detect_lof(X, n_neighbors=20, contamination=0.05):
    """LOF局部异常因子检测"""
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    pred = lof.fit_predict(X)
    labels = np.where(pred == -1, 1, 0)
    scores = -lof.negative_outlier_factor_
    print(f"LOF检测: k={n_neighbors}, 异常数={labels.sum()}/{len(X)}")
    return labels, scores, lof


# ============================================================
# 5. One-Class SVM
# ============================================================
def detect_ocsvm(X, nu=0.05, kernel='rbf'):
    """One-Class SVM异常检测"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    ocsvm = OneClassSVM(nu=nu, kernel=kernel, gamma='scale')
    pred = ocsvm.fit_predict(X_scaled)
    labels = np.where(pred == -1, 1, 0)
    print(f"One-Class SVM检测: nu={nu}, 异常数={labels.sum()}/{len(X)}")
    return labels, ocsvm


# ============================================================
# 6. 基于PCA重构误差的异常检测
# ============================================================
def detect_pca_reconstruction(X, n_components=0.95, threshold_percentile=95):
    """PCA重构误差异常检测"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X_scaled)
    X_reconstructed = pca.inverse_transform(X_reduced)

    # 重构误差
    reconstruction_error = np.sum((X_scaled - X_reconstructed) ** 2, axis=1)
    threshold = np.percentile(reconstruction_error, threshold_percentile)
    labels = np.where(reconstruction_error > threshold, 1, 0)

    print(f"PCA重构误差检测: 成分数={X_reduced.shape[1]}, "
          f"阈值百分位={threshold_percentile}, 异常数={labels.sum()}/{len(X)}")

    # 可视化重构误差分布
    plt.figure(figsize=(8, 4))
    plt.hist(reconstruction_error, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(threshold, color='r', linestyle='--', label=f'阈值 ({threshold:.2f})')
    plt.xlabel('重构误差')
    plt.ylabel('频数')
    plt.title('PCA重构误差分布')
    plt.legend()
    plt.savefig('PCA重构误差分布.png', dpi=150, bbox_inches='tight')
    plt.show()

    return labels, reconstruction_error, pca


# ============================================================
# 7. 方法对比
# ============================================================
def compare_anomaly_methods():
    """对比多种异常检测方法"""
    X, y_true = create_anomaly_data(n_normal=300, n_outliers=20)

    methods = {
        'Z-Score': lambda: detect_zscore(X, threshold=3),
        'IQR': lambda: detect_iqr(X, factor=1.5),
        '孤立森林': lambda: detect_isolation_forest(X, contamination=0.06)[0],
        'LOF': lambda: detect_lof(X, n_neighbors=20, contamination=0.06)[0],
        'One-Class SVM': lambda: detect_ocsvm(X, nu=0.06)[0],
    }

    print("=" * 50)
    print("异常检测方法对比")
    print("=" * 50)
    print(f"真实异常数: {int(y_true.sum())}/{len(y_true)}")

    for name, method_fn in methods.items():
        labels = method_fn()
        # 计算检测准确度
        tp = np.sum((labels == 1) & (y_true == 1))
        fp = np.sum((labels == 1) & (y_true == 0))
        fn = np.sum((labels == 0) & (y_true == 1))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"  {name:15s} | 检出:{labels.sum():3d} | P={precision:.3f} R={recall:.3f} F1={f1:.3f}")

    # 可视化对比
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes_flat = axes.flatten()

    for i, (name, method_fn) in enumerate(methods.items()):
        labels = method_fn()
        ax = axes_flat[i]
        mask_normal = labels == 0
        mask_anomaly = labels == 1
        ax.scatter(X[mask_normal, 0], X[mask_normal, 1], c='blue', s=15, alpha=0.5)
        ax.scatter(X[mask_anomaly, 0], X[mask_anomaly, 1], c='red', s=30, marker='x')
        ax.set_title(name)

    # 真实标签
    ax = axes_flat[5]
    mask_normal = y_true == 0
    mask_anomaly = y_true == 1
    ax.scatter(X[mask_normal, 0], X[mask_normal, 1], c='blue', s=15, alpha=0.5)
    ax.scatter(X[mask_anomaly, 0], X[mask_anomaly, 1], c='red', s=30, marker='x')
    ax.set_title('真实标签')

    plt.suptitle('异常检测方法对比', fontsize=14)
    plt.tight_layout()
    plt.savefig('异常检测方法对比.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 主程序演示
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("异常检测完整流程演示")
    print("=" * 60)

    X, y_true = create_anomaly_data()

    # 1. Z-Score
    print("\n--- 1. Z-Score ---")
    labels_z = detect_zscore(X)
    plot_anomalies(X, labels_z, 'Z-Score异常检测')

    # 2. IQR
    print("\n--- 2. IQR ---")
    labels_iqr = detect_iqr(X)
    plot_anomalies(X, labels_iqr, 'IQR异常检测')

    # 3. 孤立森林
    print("\n--- 3. 孤立森林 ---")
    labels_iso, scores_iso, _ = detect_isolation_forest(X)
    plot_anomalies(X, labels_iso, '孤立森林异常检测')

    # 4. LOF
    print("\n--- 4. LOF ---")
    labels_lof, scores_lof, _ = detect_lof(X)
    plot_anomalies(X, labels_lof, 'LOF异常检测')

    # 5. One-Class SVM
    print("\n--- 5. One-Class SVM ---")
    labels_ocsvm, _ = detect_ocsvm(X)
    plot_anomalies(X, labels_ocsvm, 'One-Class SVM异常检测')

    # 6. PCA重构误差
    print("\n--- 6. PCA重构误差 ---")
    labels_pca, errors, _ = detect_pca_reconstruction(X)

    # 7. 综合对比
    print("\n--- 7. 综合对比 ---")
    compare_anomaly_methods()
