"""
数据挖掘导论模块 - Data Mining Fundamentals
============================================
涵盖数据挖掘的基础概念与方法论：
1. 数据挖掘定义与任务分类
2. CRISP-DM跨行业数据挖掘标准流程
3. 数据类型与数据质量
4. 相似度与距离度量
5. 数据挖掘应用场景
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import setup_chinese_font

import numpy as np
from numpy.typing import NDArray
from collections.abc import Iterable
import matplotlib.pyplot as plt

setup_chinese_font()


# ============================================================
# 1. 数据挖掘定义与任务分类
# ============================================================
def print_mining_tasks():
    """打印数据挖掘的任务分类体系"""
    tasks = {
        "描述性任务": {
            "特征刻画": "总结数据的一般特征或规律（如均值、分布）",
            "关联分析": "发现数据项之间的关联规则（如购物篮分析）",
            "聚类分析": "将数据对象分组，使组内相似、组间相异",
            "异常检测": "识别偏离正常行为的数据对象",
        },
        "预测性任务": {
            "分类": "根据已有标签预测新数据的类别（如垃圾邮件识别）",
            "回归": "预测连续值输出（如房价预测）",
            "时间序列分析": "基于历史数据预测未来趋势",
            "序列模式挖掘": "发现时间序列中的频繁模式",
        },
    }

    print("=" * 60)
    print("数据挖掘任务分类体系")
    print("=" * 60)
    for category, subtasks in tasks.items():
        print(f"\n【{category}】")
        for name, desc in subtasks.items():
            print(f"  • {name}：{desc}")
    print()


# ============================================================
# 2. CRISP-DM 标准流程
# ============================================================
def print_crisp_dm():
    """展示 CRISP-DM 跨行业数据挖掘标准流程"""
    phases = [
        ("1. 业务理解", "Business Understanding",
         "明确业务目标、评估资源、确定数据挖掘目标、生成项目计划"),
        ("2. 数据理解", "Data Understanding",
         "收集初始数据、描述数据、探索数据、验证数据质量"),
        ("3. 数据准备", "Data Preparation",
         "数据选择、数据清洗、数据构造、数据整合、数据格式化"),
        ("4. 建立模型", "Modeling",
         "选择建模技术、生成测试设计、构建模型、评估模型"),
        ("5. 模型评估", "Evaluation",
         "评估挖掘结果、回顾过程、确定后续步骤"),
        ("6. 部署应用", "Deployment",
         "部署计划、监控维护、最终报告、项目回顾"),
    ]

    print("=" * 60)
    print("CRISP-DM 跨行业数据挖掘标准流程")
    print("=" * 60)
    for cn, en, desc in phases:
        print(f"\n{cn} ({en})")
        print(f"  关键活动：{desc}")
    print()
    print("注意：CRISP-DM 不是线性流程，各阶段之间需要反复迭代。")


# ============================================================
# 3. 数据类型与数据质量
# ============================================================
def data_types_overview():
    """展示数据类型体系"""
    data_types = {
        "属性类型": {
            "标称属性 (Nominal)": "无序类别，如性别、颜色",
            "有序属性 (Ordinal)": "有序但无等距，如学历、满意度",
            "区间属性 (Interval)": "有序等距但无绝对零点，如温度(℃)",
            "比率属性 (Ratio)": "有序等距且有绝对零点，如身高、收入",
        },
        "数据集类型": {
            "记录数据": "数据矩阵、事务数据、文档-词矩阵",
            "图数据": "社交网络、分子结构、Web图",
            "有序数据": "时间序列、序列数据、空间数据",
        },
        "数据质量问题": [
            "缺失值 (Missing Values)",
            "噪声与异常值 (Noise & Outliers)",
            "重复数据 (Duplicate Records)",
            "不一致数据 (Inconsistent Data)",
        ],
    }

    print("=" * 60)
    print("数据类型与数据质量")
    print("=" * 60)
    for category, items in data_types.items():
        print(f"\n【{category}】")
        if isinstance(items, dict):
            for k, v in items.items():
                print(f"  • {k}：{v}")
        else:
            for item in items:
                print(f"  • {item}")


# ============================================================
# 4. 相似度与距离度量
# ============================================================
def euclidean_distance(x: NDArray, y: NDArray) -> float:
    """欧氏距离 — 最常用的距离度量"""
    if x.shape != y.shape:
        raise ValueError(f"输入向量维度不一致: x 的维度为 {x.shape}, y 的维度为 {y.shape}")
    return np.sqrt(np.sum((x - y) ** 2))


def manhattan_distance(x: NDArray, y: NDArray) -> float:
    """曼哈顿距离 (L1范数)"""
    if x.shape != y.shape:
        raise ValueError(f"输入向量维度不一致: x 的维度为 {x.shape}, y 的维度为 {y.shape}")
    return np.sum(np.abs(x - y))


def minkowski_distance(x: NDArray, y: NDArray, p: float = 2) -> float:
    """闵可夫斯基距离 — 欧氏和曼哈顿的推广"""
    if x.shape != y.shape:
        raise ValueError(f"输入向量维度不一致: x 的维度为 {x.shape}, y 的维度为 {y.shape}")
    return np.power(np.sum(np.abs(x - y) ** p), 1 / p)


def cosine_similarity(x: NDArray, y: NDArray) -> float:
    """余弦相似度 — 衡量方向相似性，常用于文本和推荐系统"""
    if x.shape != y.shape:
        raise ValueError(f"输入向量维度不一致: x 的维度为 {x.shape}, y 的维度为 {y.shape}")
    dot = np.dot(x, y)
    norm = np.linalg.norm(x) * np.linalg.norm(y)
    return dot / norm if norm != 0 else 0


def jaccard_similarity(A: Iterable, B: Iterable) -> float:
    """Jaccard相似度 — 用于集合比较，常用于推荐和文本"""
    A, B = set(A), set(B)
    intersection = len(A & B)
    union = len(A | B)
    return intersection / union if union != 0 else 0


def pearson_correlation(x: NDArray, y: NDArray) -> float:
    """Pearson相关系数 — 衡量线性相关程度"""
    if x.shape != y.shape:
        raise ValueError(f"输入向量长度不一致: x 的长度为 {len(x)}, y 的长度为 {len(y)}")
    if len(x) < 2:
        raise ValueError(f"输入向量长度必须大于等于2，当前长度为 {len(x)}")
    n = len(x)
    mean_x, mean_y = np.mean(x), np.mean(y)
    cov = np.sum((x - mean_x) * (y - mean_y)) / (n - 1)
    std_x = np.std(x, ddof=1)
    std_y = np.std(y, ddof=1)
    return cov / (std_x * std_y) if std_x * std_y != 0 else 0


def demo_distance_metrics():
    """演示各种距离/相似度度量"""
    print("=" * 60)
    print("相似度与距离度量示例")
    print("=" * 60)

    # 数值向量
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 3, 4, 5, 6])

    print(f"\n向量 x = {x}, y = {y}")
    print(f"  欧氏距离:      {euclidean_distance(x, y):.4f}")
    print(f"  曼哈顿距离:    {manhattan_distance(x, y):.4f}")
    print(f"  闵可夫斯基(p=3): {minkowski_distance(x, y, p=3):.4f}")
    print(f"  余弦相似度:    {cosine_similarity(x, y):.4f}")
    print(f"  Pearson相关:   {pearson_correlation(x, y):.4f}")

    # 集合
    A = [1, 2, 3, 4]
    B = [3, 4, 5, 6]
    print(f"\n集合 A = {A}, B = {B}")
    print(f"  Jaccard相似度: {jaccard_similarity(A, B):.4f}")


# ============================================================
# 5. 数据挖掘应用场景
# ============================================================
def print_applications():
    """展示数据挖掘的典型应用场景"""
    applications = {
        "商业与营销": [
            "客户细分与画像",
            "购物篮分析（关联规则）",
            "推荐系统（协同过滤、内容推荐）",
            "客户流失预测",
            "精准营销与定价优化",
        ],
        "金融与风控": [
            "信用评分与欺诈检测",
            "股市预测与量化交易",
            "反洗钱与异常交易检测",
        ],
        "医疗与健康": [
            "疾病预测与辅助诊断",
            "药物发现与基因组分析",
            "医疗影像分析",
        ],
        "互联网与社交": [
            "搜索引擎与网页排名(PageRank)",
            "社交网络分析与社区发现",
            "舆情分析与情感分析",
        ],
        "工业与物联网": [
            "设备故障预测与健康管理",
            "质量控制与异常检测",
            "供应链优化",
        ],
    }

    print("=" * 60)
    print("数据挖掘典型应用场景")
    print("=" * 60)
    for domain, apps in applications.items():
        print(f"\n【{domain}】")
        for app in apps:
            print(f"  • {app}")


def visualize_distance_metrics():
    np.random.seed(42)
    n_points = 6
    points = np.random.rand(n_points, 2) * 10
    ref_idx = 0
    ref_point = points[ref_idx]

    metric_names = ['欧氏距离', '曼哈顿距离', '闵可夫斯基(p=3)', '余弦距离', 'Pearson距离']
    distances = []
    for i in range(n_points):
        if i == ref_idx:
            distances.append([0] * 5)
            continue
        d_euc = euclidean_distance(ref_point, points[i])
        d_man = manhattan_distance(ref_point, points[i])
        d_min = minkowski_distance(ref_point, points[i], p=3)
        d_cos = 1 - cosine_similarity(ref_point, points[i])
        d_prs = 1 - pearson_correlation(ref_point, points[i])
        distances.append([d_euc, d_man, d_min, d_cos, d_prs])

    distances = np.array(distances)
    labels = [f'P{i}' for i in range(n_points)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.scatter(points[:, 0], points[:, 1], c='steelblue', s=100, edgecolors='k', zorder=3)
    for i, (x, y) in enumerate(points):
        ax1.annotate(labels[i], (x, y), textcoords="offset points", xytext=(5, 5), fontsize=10)
    ax1.scatter(ref_point[0], ref_point[1], c='red', s=150, edgecolors='k', marker='*', zorder=4, label='参考点P0')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('2D散点数据')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    x_pos = np.arange(n_points - 1)
    width = 0.15
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    for j, (metric, color) in enumerate(zip(metric_names, colors)):
        vals = distances[distances[:, 0] != 0][:, j] if j == 0 else [distances[i][j] for i in range(n_points) if i != ref_idx]
        ax2.bar(x_pos + j * width, vals, width, label=metric, color=color, alpha=0.85)
    ax2.set_xlabel('数据点')
    ax2.set_ylabel('距离/相异度')
    ax2.set_title('距离度量对比')
    ax2.set_xticks(x_pos + width * 2)
    ax2.set_xticklabels([labels[i] for i in range(n_points) if i != ref_idx])
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()


# ============================================================
# 主函数
# ============================================================
if __name__ == '__main__':
    print_mining_tasks()
    print()
    print_crisp_dm()
    print()
    data_types_overview()
    print()
    demo_distance_metrics()
    print()
    print_applications()
    print()
    visualize_distance_metrics()
