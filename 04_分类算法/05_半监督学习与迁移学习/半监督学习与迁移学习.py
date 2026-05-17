"""
半监督学习与迁移学习模块 - Semi-Supervised & Transfer Learning
=============================================================
涵盖标签稀缺场景下的学习策略：
1. 半监督学习概述与假设
2. 自训练 (Self-Training)
3. 协同训练 (Co-Training)
4. 标签传播 (Label Propagation)
5. 迁移学习 (Transfer Learning)

参考：Han & Kamber《数据挖掘：概念与技术》第9.7节
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import numpy as np
from sklearn.datasets import make_moons, make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
from utils import setup_chinese_font

setup_chinese_font()


# ============================================================
# 1. 半监督学习概述
# ============================================================
def introduce_semi_supervised_learning():
    """介绍半监督学习的背景与假设"""

    print("=" * 60)
    print("半监督学习概述")
    print("=" * 60)

    print("""
    核心动机：标注数据稀缺，未标注数据丰富

    数据场景：
    ┌─────────────────────────────────────────────┐
    │  标注数据 (少量)  │  未标注数据 (大量)         │
    │  (x₁,y₁)...(xₗ,yₗ) │  xₗ₊₁...xₗ₊ᵤ  (l << u) │
    └─────────────────────────────────────────────┘

    三大基本假设：
    1. 平滑假设：相近的数据点倾向于有相同的标签
    2. 聚类假设：同一聚类中的数据点倾向于有相同的标签
    3. 流形假设：高维数据分布在低维流形上

    主要方法：
    · 自训练 (Self-Training): 用自身预测结果扩展训练集
    · 协同训练 (Co-Training): 两个分类器互相提供伪标签
    · 标签传播 (Label Propagation): 基于图的半监督方法
    · 生成式方法: 联合建模 p(x,y) 和 p(x)
    """)


# ============================================================
# 2. 自训练 (Self-Training)
# ============================================================
def self_training_demo():
    """自训练算法演示"""

    print("\n" + "=" * 60)
    print("自训练 (Self-Training)")
    print("=" * 60)

    # 生成数据
    X, y = make_moons(n_samples=500, noise=0.2, random_state=42)

    # 模拟少量标注数据
    labeled_idx = np.random.RandomState(42).choice(len(X), 20, replace=False)
    unlabeled_mask = np.ones(len(X), dtype=bool)
    unlabeled_mask[labeled_idx] = False

    X_labeled = X[labeled_idx]
    y_labeled = y[labeled_idx]
    X_unlabeled = X[unlabeled_mask]

    print(f"标注数据: {len(X_labeled)}, 未标注数据: {len(X_unlabeled)}")

    # 仅用标注数据训练（基线）
    baseline_clf = SVC(kernel='rbf', probability=True, random_state=42)
    baseline_clf.fit(X_labeled, y_labeled)
    baseline_acc = accuracy_score(y[unlabeled_mask], baseline_clf.predict(X_unlabeled))
    print(f"\n基线(仅标注数据): 准确率={baseline_acc:.3f}")

    # 自训练过程
    clf = SVC(kernel='rbf', probability=True, random_state=42)
    X_train = X_labeled.copy()
    y_train = y_labeled.copy()
    X_pool = X_unlabeled.copy()

    for round_num in range(5):
        clf.fit(X_train, y_train)

        if len(X_pool) == 0:
            break

        # 对未标注数据预测
        probs = clf.predict_proba(X_pool)
        max_probs = np.max(probs, axis=1)
        predictions = clf.predict(X_pool)

        # 选择高置信度样本（阈值=0.95）
        threshold = 0.95
        confident_mask = max_probs >= threshold
        n_confident = np.sum(confident_mask)

        if n_confident == 0:
            print(f"  第{round_num+1}轮: 无高置信度样本，停止")
            break

        # 将高置信度样本加入训练集（伪标签）
        X_train = np.vstack([X_train, X_pool[confident_mask]])
        y_train = np.concatenate([y_train, predictions[confident_mask]])

        # 从池中移除
        X_pool = X_pool[~confident_mask]

        # 计算当前准确率
        current_acc = accuracy_score(y[unlabeled_mask], clf.predict(X_unlabeled))
        print(f"  第{round_num+1}轮: 添加{n_confident}个伪标签, "
              f"训练集={len(X_train)}, 准确率={current_acc:.3f}")

    final_acc = accuracy_score(y[unlabeled_mask], clf.predict(X_unlabeled))
    print(f"\n自训练最终: 准确率={final_acc:.3f} (提升={final_acc-baseline_acc:+.3f})")

    return X, y, labeled_idx, clf, baseline_clf


# ============================================================
# 3. 协同训练 (Co-Training)
# ============================================================
def co_training_demo():
    """协同训练算法演示"""

    print("\n" + "=" * 60)
    print("协同训练 (Co-Training)")
    print("=" * 60)

    # 生成数据 - 两个独立视图
    X, y = make_classification(
        n_samples=300, n_features=20, n_informative=10,
        n_redundant=5, random_state=42
    )

    # 划分两个视图（特征子集）
    view1_features = list(range(0, 10))
    view2_features = list(range(10, 20))

    # 少量标注
    labeled_idx = np.random.RandomState(42).choice(len(X), 20, replace=False)
    unlabeled_mask = np.ones(len(X), dtype=bool)
    unlabeled_mask[labeled_idx] = False

    X_labeled = X[labeled_idx]
    y_labeled = y[labeled_idx]
    X_unlabeled = X[unlabeled_mask]
    y_unlabeled_true = y[unlabeled_mask]

    print(f"标注数据: {len(X_labeled)}, 未标注数据: {len(X_unlabeled)}")

    # 两个分类器
    h1 = SVC(kernel='rbf', probability=True, random_state=42)
    h2 = SVC(kernel='rbf', probability=True, random_state=42)

    X1_train = X_labeled[:, view1_features].copy()
    X2_train = X_labeled[:, view2_features].copy()
    y1_train = y_labeled.copy()
    y2_train = y_labeled.copy()

    X1_pool = X_unlabeled[:, view1_features].copy()
    X2_pool = X_unlabeled[:, view2_features].copy()

    for round_num in range(5):
        h1.fit(X1_train, y1_train)
        h2.fit(X2_train, y2_train)

        if len(X1_pool) == 0:
            break

        # h1为h2提供伪标签
        probs1 = h1.predict_proba(X1_pool)
        max_probs1 = np.max(probs1, axis=1)
        preds1 = h1.predict(X1_pool)

        # h2为h1提供伪标签
        probs2 = h2.predict_proba(X2_pool)
        max_probs2 = np.max(probs2, axis=1)
        preds2 = h2.predict(X2_pool)

        # 各选top-N个高置信度样本
        N = min(5, len(X1_pool))
        top1 = np.argsort(max_probs1)[-N:]
        top2 = np.argsort(max_probs2)[-N:]

        # 交叉添加伪标签
        X2_train = np.vstack([X2_train, X2_pool[top1]])
        y2_train = np.concatenate([y2_train, preds1[top1]])

        X1_train = np.vstack([X1_train, X1_pool[top2]])
        y1_train = np.concatenate([y1_train, preds2[top2]])

        # 从池中移除已标注的
        remove_idx = set(top1) | set(top2)
        keep_mask = np.ones(len(X1_pool), dtype=bool)
        for idx in remove_idx:
            keep_mask[idx] = False

        X1_pool = X1_pool[keep_mask]
        X2_pool = X2_pool[keep_mask]

        # 评估
        combined_pred = (h1.predict(X_unlabeled[:, view1_features]) +
                        h2.predict(X_unlabeled[:, view2_features])) >= 1
        acc = accuracy_score(y_unlabeled_true, combined_pred.astype(int))
        print(f"  第{round_num+1}轮: 训练集1={len(X1_train)}, "
              f"训练集2={len(X2_train)}, 准确率={acc:.3f}")


# ============================================================
# 4. 标签传播 (Label Propagation)
# ============================================================
def label_propagation_demo():
    """标签传播算法演示"""

    print("\n" + "=" * 60)
    print("标签传播 (Label Propagation)")
    print("=" * 60)

    # 生成数据
    np.random.seed(42)
    X, y = make_moons(n_samples=300, noise=0.15, random_state=42)

    # 少量标注
    labeled_idx = np.random.RandomState(42).choice(len(X), 10, replace=False)
    y_input = np.full(len(X), -1)  # -1表示未标注
    y_input[labeled_idx] = y[labeled_idx]

    print(f"标注数据: {len(labeled_idx)}, 未标注数据: {np.sum(y_input == -1)}")

    # 构建相似度矩阵（RBF核）
    sigma = 0.3
    n = len(X)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist = np.sum((X[i] - X[j]) ** 2)
            W[i, j] = np.exp(-dist / (2 * sigma ** 2))

    # 归一化
    D = np.diag(W.sum(axis=1))
    D_inv = np.linalg.inv(D)
    S = D_inv @ W  # 转移概率矩阵

    # 迭代传播
    n_classes = 2
    F = np.zeros((n, n_classes))
    for i in range(n):
        if y_input[i] >= 0:
            F[i, y_input[i]] = 1.0

    alpha = 0.99  # 传播系数

    for iteration in range(50):
        F_new = alpha * S @ F + (1 - alpha) * F
        # 固定已标注节点的标签
        for i in range(n):
            if y_input[i] >= 0:
                F_new[i] = 0
                F_new[i, y_input[i]] = 1.0
        F = F_new

    # 预测
    y_pred = np.argmax(F, axis=1)
    acc = accuracy_score(y, y_pred)
    print(f"标签传播准确率: {acc:.3f}")

    return X, y, labeled_idx, y_pred


# ============================================================
# 5. 迁移学习 (Transfer Learning)
# ============================================================
def transfer_learning_demo():
    """迁移学习演示"""

    print("\n" + "=" * 60)
    print("迁移学习 (Transfer Learning)")
    print("=" * 60)

    print("""
    核心思想：将源域(Ds)学到的知识迁移到目标域(Dt)

    场景分类：
    ┌──────────┬───────────────────┬───────────────────┐
    │          │ 源域和目标域特征相同   │ 源域和目标域特征不同  │
    ├──────────┼───────────────────┼───────────────────┤
    │ 域相同    │ 归纳迁移学习        │ 异构迁移学习        │
    │ 任务相同   │ (多任务学习)        │ (特征映射)         │
    ├──────────┼───────────────────┼───────────────────┤
    │ 域不同    │ 直推迁移学习        │ 无监督迁移学习      │
    │ 任务相同   │ (域自适应)         │ (知识蒸馏)         │
    └──────────┴───────────────────┴───────────────────┘

    主要方法：
    1. 基于实例：重加权源域样本 (TrAdaBoost)
    2. 基于特征：学习域不变特征表示 (DAN, CORAL)
    3. 基于模型：预训练+微调 (Fine-tuning)
    4. 基于关系：迁移关系知识 (图谱迁移)
    """)

    # 演示：特征迁移（微调方式）
    print("示例：预训练+微调的迁移策略")
    print("-" * 40)

    # 源域数据（数据量大）
    X_source, y_source = make_classification(
        n_samples=1000, n_features=20, n_informative=10,
        random_state=42
    )

    # 目标域数据（数据量小，分布偏移）
    X_target, y_target = make_classification(
        n_samples=50, n_features=20, n_informative=10,
        random_state=123  # 不同分布
    )

    # 策略1：仅用目标域数据
    clf_target_only = SVC(kernel='rbf', random_state=42)
    clf_target_only.fit(X_target, y_target)
    acc_target_only = accuracy_score(y_target, clf_target_only.predict(X_target))

    # 策略2：用源域预训练（模拟）
    clf_source = SVC(kernel='rbf', random_state=42)
    clf_source.fit(X_source, y_source)

    # 策略3：源域+目标域混合训练
    X_mixed = np.vstack([X_source, X_target])
    y_mixed = np.concatenate([y_source, y_target])
    clf_mixed = SVC(kernel='rbf', random_state=42)
    clf_mixed.fit(X_mixed, y_mixed)
    acc_mixed = accuracy_score(y_target, clf_mixed.predict(X_target))

    print(f"仅目标域数据(50样本): 准确率={acc_target_only:.3f}")
    print(f"源域预训练(1000样本): 准确率={accuracy_score(y_target, clf_source.predict(X_target)):.3f}")
    print(f"源+目标混合训练:      准确率={acc_mixed:.3f}")

    # TrAdaBoost思想说明
    print("""
    TrAdaBoost核心思想：
    · 同时使用源域和目标域数据训练
    · 源域中被错误分类的样本 → 降低权重（可能不适用于目标域）
    · 目标域中被错误分类的样本 → 提高权重（需要更多关注）
    · 迭代调整权重，使模型逐渐聚焦目标域
    """)


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════╗")
    print("║  半监督学习与迁移学习                         ║")
    print("║  Semi-Supervised & Transfer Learning         ║")
    print("╚══════════════════════════════════════════════╝")

    introduce_semi_supervised_learning()
    self_training_demo()
    co_training_demo()
    label_propagation_demo()
    transfer_learning_demo()
