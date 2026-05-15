"""
流数据挖掘模块 - Stream Data Mining
=====================================
涵盖流数据挖掘的核心概念与方法：
1. 流数据特征与挑战
2. 滑动窗口模型
3. 衰减因子机制
4. 在线学习与概念漂移检测
5. 流聚类 (CluStream)

参考：ICDM 2025、DaKM 2025专题
"""

import numpy as np
from collections import deque, defaultdict

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 流数据特征与挑战
# ============================================================
def introduce_stream_mining():
    """介绍流数据挖掘的背景与挑战"""

    print("=" * 60)
    print("流数据挖掘概述")
    print("=" * 60)

    characteristics = {
        "数据特征": {
            "无限流": "数据持续到达，总量无界",
            "高速率": "到达速度快，无法存储全部数据",
            "时变性": "数据分布可能随时间变化(概念漂移)",
            "不可重复": "数据流通常只能单次扫描",
        },
        "与传统挖掘的区别": {
            "存储": "传统→全部存储 | 流→仅存储摘要/窗口",
            "扫描": "传统→多次扫描 | 流→单次扫描",
            "算法": "传统→批量处理 | 流→增量/在线更新",
            "模型": "传统→静态模型 | 流→动态自适应模型",
        },
        "典型应用": {
            "网络监控": "入侵检测、流量异常",
            "金融交易": "欺诈检测、实时风控",
            "传感器网络": "IoT数据实时分析",
            "社交媒体": "热点话题追踪、舆情监控",
            "电商推荐": "用户行为实时反馈",
        },
    }

    for category, items in characteristics.items():
        print(f"\n【{category}】")
        for key, value in items.items():
            print(f"  · {key}: {value}")


# ============================================================
# 2. 滑动窗口模型
# ============================================================
class SlidingWindow:
    """
    滑动窗口模型：只保留最近N个数据点

    三种窗口类型：
    - 固定窗口：保持固定大小N
    - 衰减窗口：旧数据权重指数衰减
    - 自适应窗口：根据变化率动态调整大小
    """

    def __init__(self, window_size=100):
        self.window_size = window_size
        self.window = deque(maxlen=window_size)

    def add(self, value):
        """添加新数据点，自动淘汰最旧数据"""
        self.window.append(value)

    def get_statistics(self):
        """计算窗口内统计量"""
        if not self.window:
            return {}
        data = np.array(self.window)
        return {
            'count': len(data),
            'mean': np.mean(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data),
        }

    def detect_anomaly(self, value, n_sigma=3):
        """基于窗口统计的异常检测"""
        stats = self.get_statistics()
        if stats.get('count', 0) < 10:
            return False, "窗口数据不足"

        z_score = (value - stats['mean']) / (stats['std'] + 1e-8)
        is_anomaly = abs(z_score) > n_sigma

        return is_anomaly, f"z-score={z_score:.2f}"


def demonstrate_sliding_window():
    """演示滑动窗口模型"""

    print("\n" + "=" * 60)
    print("滑动窗口模型")
    print("=" * 60)

    np.random.seed(42)
    window = SlidingWindow(window_size=50)

    # 模拟正常数据流 + 突发异常
    normal_data = np.random.normal(100, 10, 80)
    anomaly_data = np.array([200, 210, 180])  # 突发异常
    data_stream = np.concatenate([normal_data[:40], anomaly_data, normal_data[40:]])

    print(f"\n窗口大小: {window.window_size}")
    print("\n逐点检测（仅显示异常点）:")

    anomalies = []
    for i, value in enumerate(data_stream):
        is_anomaly, info = window.detect_anomaly(value)
        window.add(value)
        if is_anomaly:
            anomalies.append((i, value))
            print(f"  时刻{i:3d}: 值={value:.1f}, {info} ← 异常!")

    if not anomalies:
        print("  未检测到异常")
    else:
        print(f"\n共检测到 {len(anomalies)} 个异常点")

    # 可视化
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(data_stream, 'b-', alpha=0.6, label='数据流')
    if anomalies:
        ax.scatter([a[0] for a in anomalies], [a[1] for a in anomalies],
                  c='red', s=100, zorder=5, label='异常点')
    ax.set_xlabel('时间步')
    ax.set_ylabel('值')
    ax.set_title('滑动窗口异常检测')
    ax.legend()
    plt.tight_layout()
    plt.savefig('滑动窗口异常检测.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n图表已保存: 滑动窗口异常检测.png")


# ============================================================
# 3. 衰减因子机制
# ============================================================
class FadingSummarizer:
    """
    衰减因子摘要：旧数据权重随时间指数衰减

    公式: weight(t) = (1-λ)^(current_time - t)
    λ越小，历史数据衰减越慢；λ越大，越重视近期数据
    """

    def __init__(self, decay_factor=0.01):
        self.decay_factor = decay_factor
        self.sum = 0.0
        self.sum_sq = 0.0
        self.count = 0.0  # 加权计数

    def add(self, value):
        """添加数据点并衰减历史"""
        # 衰减历史
        fade = 1 - self.decay_factor
        self.sum *= fade
        self.sum_sq *= fade
        self.count *= fade

        # 加入新值
        self.sum += value
        self.sum_sq += value ** 2
        self.count += 1

    def get_mean(self):
        """获取加权均值"""
        return self.sum / self.count if self.count > 0 else 0

    def get_variance(self):
        """获取加权方差"""
        if self.count <= 1:
            return 0
        mean = self.get_mean()
        return self.sum_sq / self.count - mean ** 2


def demonstrate_fading():
    """演示衰减因子机制"""

    print("\n" + "=" * 60)
    print("衰减因子机制")
    print("=" * 60)

    np.random.seed(42)

    # 模拟概念漂移：均值从100逐渐漂移到150
    n_points = 200
    data_stream = np.random.normal(
        loc=np.linspace(100, 150, n_points),
        scale=10
    )

    # 不同衰减因子对比
    summarizers = {
        'λ=0.005(慢衰减)': FadingSummarizer(decay_factor=0.005),
        'λ=0.02(中衰减)': FadingSummarizer(decay_factor=0.02),
        'λ=0.1(快衰减)': FadingSummarizer(decay_factor=0.1),
    }

    means = {name: [] for name in summarizers}

    for value in data_stream:
        for name, summarizer in summarizers.items():
            summarizer.add(value)
            means[name].append(summarizer.get_mean())

    # 可视化
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(data_stream, 'b.', alpha=0.2, markersize=2, label='原始数据')
    for name, mean_values in means.items():
        ax.plot(mean_values, linewidth=2, label=name)
    ax.set_xlabel('时间步')
    ax.set_ylabel('值')
    ax.set_title('衰减因子对比 - 概念漂移追踪')
    ax.legend()
    plt.tight_layout()
    plt.savefig('衰减因子对比.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("图表已保存: 衰减因子对比.png")

    # 打印最终统计
    print("\n最终加权均值(真实值≈150):")
    for name, summarizer in summarizers.items():
        print(f"  {name}: {summarizer.get_mean():.1f}")


# ============================================================
# 4. 概念漂移检测 - DDM方法
# ============================================================
def ddm_concept_drift_detection():
    """
    DDM (Drift Detection Method) 概念漂移检测

    核心思想：监控分类错误率
    - 错误率 p_i 和标准差 s_i = sqrt(p_i*(1-p_i)/i)
    - 若 p_i + s_i > p_min + 2*s_min → 警告(WARNING)
    - 若 p_i + s_i > p_min + 3*s_min → 漂移(DRIFT)
    """

    print("\n" + "=" * 60)
    print("概念漂移检测 - DDM方法")
    print("=" * 60)

    np.random.seed(42)

    # 模拟：前100步错误率0.2，后100步错误率0.5（概念漂移）
    n_steps = 200
    error_rate = np.concatenate([
        np.random.binomial(1, 0.2, 100),
        np.random.binomial(1, 0.5, 100),
    ])

    p_min = float('inf')
    s_min = float('inf')
    p_i = 0.0
    status = "NORMAL"

    for i in range(1, n_steps + 1):
        p_i = np.mean(error_rate[:i])
        s_i = np.sqrt(p_i * (1 - p_i) / i)

        if p_i + s_i < p_min + s_min:
            p_min = p_i
            s_min = s_i

        if p_i + s_i >= p_min + 3 * s_min:
            if status != "DRIFT":
                print(f"  时刻 {i}: ⚠ DRIFT检测! "
                      f"p_i={p_i:.3f}, 阈值={p_min + 3*s_min:.3f}")
                status = "DRIFT"
        elif p_i + s_i >= p_min + 2 * s_min:
            if status == "NORMAL":
                print(f"  时刻 {i}: ⚡ WARNING "
                      f"p_i={p_i:.3f}, 阈值={p_min + 2*s_min:.3f}")
                status = "WARNING"
        else:
            if status != "NORMAL":
                status = "NORMAL"


# ============================================================
# 5. 在线K-Means (流聚类简化版)
# ============================================================
class OnlineKMeans:
    """
    在线K-Means：逐点更新的流聚类算法

    核心思想：每到来一个数据点，将其分配到最近聚类中心，
    然后以学习率α更新该中心
    """

    def __init__(self, n_clusters=3, learning_rate=0.1):
        self.n_clusters = n_clusters
        self.learning_rate = learning_rate
        self.centers = None
        self.counts = np.zeros(n_clusters)

    def partial_fit(self, point):
        """单点在线更新"""
        point = np.array(point)

        if self.centers is None:
            # 初始化：第一个点作为第一个中心
            self.centers = point.reshape(1, -1)
            return 0

        if len(self.centers) < self.n_clusters:
            # 中心数不足时，将较远的点作为新中心
            dists = np.linalg.norm(self.centers - point, axis=1)
            if np.min(dists) > 1.0:
                self.centers = np.vstack([self.centers, point])
                return len(self.centers) - 1
            else:
                label = np.argmin(dists)
        else:
            # 分配到最近中心
            dists = np.linalg.norm(self.centers - point, axis=1)
            label = np.argmin(dists)

            # 在线更新中心
            alpha = self.learning_rate / (1 + self.counts[label] * 0.01)
            self.centers[label] += alpha * (point - self.centers[label])

        self.counts[label] += 1
        return label


def demonstrate_online_kmeans():
    """演示在线K-Means流聚类"""

    print("\n" + "=" * 60)
    print("在线K-Means流聚类")
    print("=" * 60)

    np.random.seed(42)

    # 生成3个簇的数据流
    cluster1 = np.random.normal([2, 2], 0.5, (50, 2))
    cluster2 = np.random.normal([6, 6], 0.5, (50, 2))
    cluster3 = np.random.normal([2, 6], 0.5, (50, 2))
    data_stream = np.vstack([cluster1, cluster2, cluster3])
    np.random.shuffle(data_stream)

    model = OnlineKMeans(n_clusters=3, learning_rate=0.1)
    labels = []

    for point in data_stream:
        label = model.partial_fit(point)
        labels.append(label)

    print(f"\n数据流大小: {len(data_stream)}")
    print(f"最终聚类中心:")
    for i, center in enumerate(model.centers):
        print(f"  簇{i}: 中心={center}, 样本数={int(model.counts[i])}")

    # 可视化
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = np.array(labels)
    colors = ['#e74c3c', '#2ecc71', '#3498db']
    for i in range(len(model.centers)):
        mask = labels == i
        ax.scatter(data_stream[mask, 0], data_stream[mask, 1],
                  c=colors[i], alpha=0.5, label=f'簇{i}')
    ax.scatter(model.centers[:, 0], model.centers[:, 1],
              c='black', marker='X', s=200, label='聚类中心')
    ax.set_title('在线K-Means流聚类结果')
    ax.legend()
    plt.tight_layout()
    plt.savefig('在线KMeans流聚类.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n图表已保存: 在线KMeans流聚类.png")


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════╗")
    print("║      流数据挖掘 - Stream Data Mining         ║")
    print("╚══════════════════════════════════════════════╝")

    introduce_stream_mining()
    demonstrate_sliding_window()
    demonstrate_fading()
    ddm_concept_drift_detection()
    demonstrate_online_kmeans()
