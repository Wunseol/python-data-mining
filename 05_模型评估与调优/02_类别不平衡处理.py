"""
类别不平衡处理模块 - Imbalanced Learning
=========================================
涵盖类别不平衡问题的处理方法：
1. 不平衡数据的问题分析
2. 过采样方法 (随机过采样、SMOTE、ADASYN)
3. 欠采样方法 (随机欠采样、Tomek Links)
4. 代价敏感学习
5. 集成方法 (EasyEnsemble、BalanceCascade)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc
)
from utils import setup_chinese_font

setup_chinese_font()


# ============================================================
# 辅助：创建不平衡数据集
# ============================================================
def create_imbalanced_data(n_samples=1000, minority_ratio=0.05, random_state=42):
    """创建不平衡二分类数据集"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_clusters_per_class=1,
        weights=[1 - minority_ratio, minority_ratio],
        random_state=random_state,
    )
    return X, y


def plot_imbalanced_data(X, y, title="不平衡数据分布"):
    """可视化不平衡数据"""
    fig, ax = plt.subplots(figsize=(8, 6))
    mask_pos = y == 1
    mask_neg = y == 0
    ax.scatter(X[mask_neg, 0], X[mask_neg, 1], c='steelblue', alpha=0.5, label=f'多数类 (n={mask_neg.sum()})')
    ax.scatter(X[mask_pos, 0], X[mask_pos, 1], c='orangered', alpha=0.7, label=f'少数类 (n={mask_pos.sum()})', marker='^')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 1. 不平衡数据的问题分析
# ============================================================
def demo_imbalance_problem():
    """演示类别不平衡导致的问题"""
    print("--- 1. 类别不平衡的问题分析 ---")
    X, y = create_imbalanced_data(minority_ratio=0.05)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # 不做任何处理，直接训练
    clf = LogisticRegression(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(f"  训练集类别分布: 多数类={sum(y_train==0)}, 少数类={sum(y_train==1)}")
    print(f"  不平衡比例: {sum(y_train==0)/sum(y_train==1):.1f}:1")
    print(f"\n  不做处理的分类结果:")
    print(f"  准确率: {accuracy_score(y_test, y_pred):.4f}")
    print(f"  少数类召回率: {recall_score(y_test, y_pred):.4f}")
    print(f"  少数类F1: {f1_score(y_test, y_pred):.4f}")
    print(f"  → 准确率很高但少数类几乎全部漏判！")


# ============================================================
# 2. 过采样方法
# ============================================================
def random_oversample(X, y, random_state=42):
    """随机过采样：复制少数类样本直到类别平衡"""
    np.random.seed(random_state)
    mask_pos = y == 1
    mask_neg = y == 0
    n_neg = mask_neg.sum()

    # 随机复制少数类样本
    pos_indices = np.where(mask_pos)[0]
    oversample_indices = np.random.choice(pos_indices, size=n_neg - mask_pos.sum(), replace=True)

    X_res = np.vstack([X, X[oversample_indices]])
    y_res = np.hstack([y, y[oversample_indices]])
    return X_res, y_res


def smote(X, y, k=5, random_state=42):
    """
    SMOTE (Synthetic Minority Oversampling Technique)
    在少数类样本之间插值生成新的合成样本
    """
    np.random.seed(random_state)
    mask_pos = y == 1
    mask_neg = y == 0

    X_pos = X[mask_pos]
    n_pos = X_pos.shape[0]
    n_neg = mask_neg.sum()
    n_synthetic = n_neg - n_pos

    # 对每个少数类样本找K近邻
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(k + 1, n_pos), metric='euclidean')
    nn.fit(X_pos)

    synthetic_samples = []
    for _ in range(n_synthetic):
        # 随机选一个少数类样本
        idx = np.random.randint(0, n_pos)
        sample = X_pos[idx]

        # 找到K近邻（排除自身）
        distances, indices = nn.kneighbors([sample])
        neighbor_idx = np.random.choice(indices[0][1:])  # 排除自身
        neighbor = X_pos[neighbor_idx]

        # 在样本和近邻之间随机插值
        diff = neighbor - sample
        gap = np.random.uniform(0, 1)
        new_sample = sample + gap * diff
        synthetic_samples.append(new_sample)

    X_synthetic = np.array(synthetic_samples)
    X_res = np.vstack([X, X_synthetic])
    y_res = np.hstack([y, np.ones(n_synthetic)])
    return X_res, y_res


def demo_oversampling():
    """演示过采样方法"""
    print("\n--- 2. 过采样方法 ---")
    X, y = create_imbalanced_data(minority_ratio=0.05)

    # 随机过采样
    X_ros, y_ros = random_oversample(X, y)
    print(f"  随机过采样后: 多数类={sum(y_ros==0)}, 少数类={sum(y_ros==1)}")

    # SMOTE
    X_smote, y_smote = smote(X, y, k=5)
    print(f"  SMOTE后:     多数类={sum(y_smote==0)}, 少数类={sum(y_smote==1)}")

    # 对比效果
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    methods = {
        "不做处理": (X_train, y_train),
        "随机过采样": random_oversample(X_train, y_train),
        "SMOTE": smote(X_train, y_train),
    }

    print(f"\n  {'方法':>12} {'准确率':>8} {'少数类召回率':>12} {'少数类F1':>10}")
    for name, (X_tr, y_tr) in methods.items():
        clf = LogisticRegression(random_state=42, max_iter=1000)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f"  {name:>12} {acc:>8.4f} {rec:>12.4f} {f1:>10.4f}")


# ============================================================
# 3. 欠采样方法
# ============================================================
def random_undersample(X, y, random_state=42):
    """随机欠采样：随机删除多数类样本"""
    np.random.seed(random_state)
    mask_pos = y == 1
    mask_neg = y == 0
    n_pos = mask_pos.sum()

    neg_indices = np.where(mask_neg)[0]
    selected_neg = np.random.choice(neg_indices, size=n_pos, replace=False)

    pos_indices = np.where(mask_pos)[0]
    selected = np.hstack([pos_indices, selected_neg])
    np.random.shuffle(selected)

    return X[selected], y[selected]


def demo_undersampling():
    """演示欠采样方法"""
    print("\n--- 3. 欠采样方法 ---")
    X, y = create_imbalanced_data(minority_ratio=0.05)

    X_rus, y_rus = random_undersample(X, y)
    print(f"  随机欠采样后: 多数类={sum(y_rus==0)}, 少数类={sum(y_rus==1)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    methods = {
        "不做处理": (X_train, y_train),
        "随机欠采样": random_undersample(X_train, y_train),
    }

    print(f"\n  {'方法':>12} {'准确率':>8} {'少数类召回率':>12} {'少数类F1':>10}")
    for name, (X_tr, y_tr) in methods.items():
        clf = LogisticRegression(random_state=42, max_iter=1000)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f"  {name:>12} {acc:>8.4f} {rec:>12.4f} {f1:>10.4f}")


# ============================================================
# 4. 代价敏感学习
# ============================================================
def demo_cost_sensitive():
    """演示代价敏感学习"""
    print("\n--- 4. 代价敏感学习 ---")
    X, y = create_imbalanced_data(minority_ratio=0.05)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # 不做处理
    clf_normal = LogisticRegression(random_state=42, max_iter=1000)
    clf_normal.fit(X_train, y_train)
    y_pred_normal = clf_normal.predict(X_test)

    # 代价敏感：给少数类更高权重
    weight_ratio = sum(y_train == 0) / sum(y_train == 1)
    clf_cost = LogisticRegression(random_state=42, max_iter=1000, class_weight={0: 1, 1: weight_ratio})
    clf_cost.fit(X_train, y_train)
    y_pred_cost = clf_cost.predict(X_test)

    # 自动balanced模式
    clf_balanced = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    clf_balanced.fit(X_train, y_train)
    y_pred_balanced = clf_balanced.predict(X_test)

    print(f"  少数类权重比例: {weight_ratio:.1f}")
    print(f"\n  {'方法':>16} {'准确率':>8} {'少数类召回率':>12} {'少数类F1':>10}")
    for name, y_pred in [("不做处理", y_pred_normal),
                         (f"代价敏感(w={weight_ratio:.0f})", y_pred_cost),
                         ("balanced模式", y_pred_balanced)]:
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f"  {name:>16} {acc:>8.4f} {rec:>12.4f} {f1:>10.4f}")


# ============================================================
# 5. 方法对比与选择建议
# ============================================================
def print_strategy_guide():
    """不平衡数据处理策略选择指南"""
    strategies = {
        "数据量充足": [
            "优先尝试 SMOTE 过采样",
            "可尝试 SMOTE + 欠采样组合",
            "代价敏感学习作为基线对比",
        ],
        "数据量不足": [
            "优先使用代价敏感学习（不损失数据）",
            "SMOTE 可能生成噪声样本，需谨慎",
            "欠采样可能丢失重要信息",
        ],
        "极端不平衡 (<1%)": [
            "将问题转化为异常检测",
            "使用 One-Class SVM 或孤立森林",
            "考虑半监督学习方法",
        ],
        "多分类不平衡": [
            "SMOTE 变体: SMOTE-ENN, SMOTE-Tomek",
            "逐类调整 class_weight",
            "使用 BalancedRandomForest",
        ],
    }

    print("\n--- 5. 不平衡数据处理策略选择 ---")
    for scenario, recommendations in strategies.items():
        print(f"\n  【{scenario}】")
        for rec in recommendations:
            print(f"    • {rec}")


# ============================================================
# 主函数
# ============================================================
if __name__ == '__main__':
    demo_imbalance_problem()
    demo_oversampling()
    demo_undersampling()
    demo_cost_sensitive()
    print_strategy_guide()
