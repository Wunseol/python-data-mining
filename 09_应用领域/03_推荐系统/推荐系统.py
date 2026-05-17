"""
推荐系统模块 - Recommender Systems
===================================
涵盖推荐系统的核心方法：
1. 协同过滤 (用户-based / 物品-based)
2. 矩阵分解推荐 (SVD)
3. 基于内容推荐
4. 推荐系统评估指标 (Precision@K, Recall@K, NDCG, MAP)
5. 冷启动问题与混合策略
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from utils import setup_chinese_font

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

setup_chinese_font()


# ============================================================
# 辅助：构建示例评分矩阵
# ============================================================
def create_rating_matrix():
    """
    创建示例用户-物品评分矩阵
    行=用户, 列=物品, 0表示未评分
    """
    return np.array([
        [5, 3, 4, 4, 0, 0, 0, 0, 0],  # 用户0
        [0, 0, 0, 0, 5, 4, 0, 0, 0],  # 用户1
        [4, 0, 5, 0, 0, 0, 3, 0, 0],  # 用户2
        [0, 0, 0, 3, 4, 5, 0, 0, 0],  # 用户3
        [0, 4, 0, 0, 0, 0, 5, 4, 3],  # 用户4
        [3, 0, 0, 0, 0, 0, 4, 5, 4],  # 用户5
    ])


# ============================================================
# 1. 基于用户的协同过滤 (User-based CF)
# ============================================================
def cosine_sim(ratings, u1, u2):
    """计算两个用户之间的余弦相似度（仅基于共同评分项）"""
    common = (ratings[u1] != 0) & (ratings[u2] != 0)
    if np.sum(common) == 0:
        return 0
    v1 = ratings[u1][common]
    v2 = ratings[u2][common]
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return dot / norm if norm != 0 else 0


def user_based_predict(ratings, user, item, k=3):
    """
    基于用户的协同过滤预测评分
    找到与目标用户最相似的K个已评分用户，加权预测
    """
    n_users = ratings.shape[0]
    sims = []
    for other in range(n_users):
        if other == user or ratings[other, item] == 0:
            continue
        sim = cosine_sim(ratings, user, other)
        if sim > 0:
            sims.append((other, sim))

    if not sims:
        return 0

    sims.sort(key=lambda x: x[1], reverse=True)
    sims = sims[:k]

    numerator = sum(sim * ratings[other, item] for other, sim in sims)
    denominator = sum(sim for _, sim in sims)
    return numerator / denominator if denominator != 0 else 0


def demo_user_based_cf():
    """演示基于用户的协同过滤"""
    print("--- 1. 基于用户的协同过滤 (User-based CF) ---")
    R = create_rating_matrix()
    print(f"评分矩阵形状: {R.shape} (6用户 × 9物品)")

    # 为用户0预测物品4的评分
    user, item = 0, 4
    pred = user_based_predict(R, user, item, k=3)
    print(f"  用户{user}对物品{item}的预测评分: {pred:.2f}")

    # 为用户0推荐Top-3物品
    recommendations = []
    for item in range(R.shape[1]):
        if R[user, item] == 0:
            pred = user_based_predict(R, user, item, k=3)
            recommendations.append((item, pred))
    recommendations.sort(key=lambda x: x[1], reverse=True)
    print(f"  用户{user}的Top-3推荐: {[(f'物品{i}', f'{s:.2f}') for i, s in recommendations[:3]]}")


# ============================================================
# 2. 基于物品的协同过滤 (Item-based CF)
# ============================================================
def item_cosine_sim(ratings, i1, i2):
    """计算两个物品之间的余弦相似度"""
    common = (ratings[:, i1] != 0) & (ratings[:, i2] != 0)
    if np.sum(common) == 0:
        return 0
    v1 = ratings[common, i1]
    v2 = ratings[common, i2]
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return dot / norm if norm != 0 else 0


def item_based_predict(ratings, user, item, k=3):
    """基于物品的协同过滤预测评分"""
    n_items = ratings.shape[1]
    sims = []
    for other_item in range(n_items):
        if other_item == item or ratings[user, other_item] == 0:
            continue
        sim = item_cosine_sim(ratings, item, other_item)
        if sim > 0:
            sims.append((other_item, sim))

    if not sims:
        return 0

    sims.sort(key=lambda x: x[1], reverse=True)
    sims = sims[:k]

    numerator = sum(sim * ratings[user, other_item] for other_item, sim in sims)
    denominator = sum(sim for _, sim in sims)
    return numerator / denominator if denominator != 0 else 0


def demo_item_based_cf():
    """演示基于物品的协同过滤"""
    print("\n--- 2. 基于物品的协同过滤 (Item-based CF) ---")
    R = create_rating_matrix()

    user, item = 0, 4
    pred = item_based_predict(R, user, item, k=3)
    print(f"  用户{user}对物品{item}的预测评分: {pred:.2f}")


# ============================================================
# 3. 矩阵分解推荐 (SVD)
# ============================================================
def svd_recommend(ratings, user, n_factors=3, n_recommend=3):
    """
    基于SVD的矩阵分解推荐
    将评分矩阵分解为用户因子矩阵和物品因子矩阵
    """
    # 构建掩码：0替换为均值（简单处理缺失值）
    mask = ratings != 0
    row_means = np.array([
        ratings[i, mask[i]].mean() if mask[i].any() else 0
        for i in range(ratings.shape[0])
    ])
    filled = ratings.copy().astype(float)
    for i in range(ratings.shape[0]):
        filled[i, ~mask[i]] = row_means[i]

    # SVD分解
    U, sigma, Vt = np.linalg.svd(filled, full_matrices=False)

    # 截断到n_factors个因子
    U_k = U[:, :n_factors]
    sigma_k = np.diag(sigma[:n_factors])
    Vt_k = Vt[:n_factors, :]

    # 重构评分矩阵
    predicted = U_k @ sigma_k @ Vt_k

    # 为目标用户推荐
    user_pred = predicted[user]
    user_rated = mask[user]
    recommendations = []
    for item in range(len(user_pred)):
        if not user_rated[item]:
            recommendations.append((item, user_pred[item]))
    recommendations.sort(key=lambda x: x[1], reverse=True)

    return recommendations[:n_recommend], predicted


def demo_svd_recommend():
    """演示SVD矩阵分解推荐"""
    print("\n--- 3. 矩阵分解推荐 (SVD) ---")
    R = create_rating_matrix()

    recs, pred_matrix = svd_recommend(R, user=0, n_factors=3, n_recommend=3)
    print(f"  用户0的Top-3推荐 (SVD): {[(f'物品{i}', f'{s:.2f}') for i, s in recs]}")


# ============================================================
# 4. 推荐系统评估指标
# ============================================================
def precision_at_k(recommended, relevant, k):
    """Precision@K：推荐列表中前K个有多少是相关的"""
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(relevant))
    return hits / k if k > 0 else 0


def recall_at_k(recommended, relevant, k):
    """Recall@K：相关物品中有多少被推荐了"""
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(relevant))
    return hits / len(relevant) if len(relevant) > 0 else 0


def ndcg_at_k(recommended, relevant, k):
    """NDCG@K：归一化折损累积增益，考虑排序位置"""
    dcg = 0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            dcg += 1 / np.log2(i + 2)  # log2(rank+1), rank从1开始
    # 理想DCG
    ideal_hits = min(len(relevant), k)
    idcg = sum(1 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0


def average_precision(recommended, relevant):
    """平均精度 (AP)：用于计算MAP"""
    hits = 0
    sum_precision = 0
    for i, item in enumerate(recommended):
        if item in relevant:
            hits += 1
            sum_precision += hits / (i + 1)
    return sum_precision / len(relevant) if len(relevant) > 0 else 0


def demo_evaluation():
    """演示推荐系统评估指标"""
    print("\n--- 4. 推荐系统评估指标 ---")
    recommended = [1, 3, 5, 7, 9, 2, 4, 6, 8, 10]
    relevant = [1, 2, 3, 5, 8, 11]

    for k in [3, 5, 10]:
        p = precision_at_k(recommended, relevant, k)
        r = recall_at_k(recommended, relevant, k)
        ndcg = ndcg_at_k(recommended, relevant, k)
        print(f"  K={k}: Precision={p:.3f}, Recall={r:.3f}, NDCG={ndcg:.3f}")

    ap = average_precision(recommended, relevant)
    print(f"  AP (Average Precision): {ap:.3f}")


# ============================================================
# 5. 冷启动问题与混合策略
# ============================================================
def print_cold_start_strategies():
    """展示冷启动问题的解决策略"""
    strategies = {
        "新用户冷启动": [
            "基于人口统计学信息推荐（年龄、性别、地域）",
            "利用社交网络关系",
            "引导用户初始评分（问卷/选择偏好）",
            "推荐热门物品",
        ],
        "新物品冷启动": [
            "基于物品内容特征推荐（文本、图像、标签）",
            "利用物品属性相似度",
            "专家标注初始评分",
        ],
        "混合策略": [
            "加权混合：多个推荐器的加权组合",
            "切换策略：根据场景切换推荐器",
            "级联：一个推荐器过滤，另一个排序",
            "特征组合：将多种特征输入同一模型",
        ],
    }

    print("\n--- 5. 冷启动问题与混合策略 ---")
    for problem, solutions in strategies.items():
        print(f"\n  【{problem}】")
        for s in solutions:
            print(f"    • {s}")


# ============================================================
# 主函数
# ============================================================
if __name__ == '__main__':
    demo_user_based_cf()
    demo_item_based_cf()
    demo_svd_recommend()
    demo_evaluation()
    print_cold_start_strategies()

    R = create_rating_matrix()
    np.random.seed(42)
    mask = R != 0
    user_based_preds = np.zeros_like(R, dtype=float)
    item_based_preds = np.zeros_like(R, dtype=float)

    for u in range(R.shape[0]):
        for i in range(R.shape[1]):
            if not mask[u, i]:
                user_based_preds[u, i] = user_based_predict(R, u, i, k=3)
                item_based_preds[u, i] = item_based_predict(R, u, i, k=3)

    ub_errors = []
    ib_errors = []
    ub_abs_errors = []
    ib_abs_errors = []
    for u in range(R.shape[0]):
        for i in range(R.shape[1]):
            if mask[u, i]:
                ub_pred = user_based_predict(R, u, i, k=3)
                ib_pred = item_based_predict(R, u, i, k=3)
                ub_errors.append((ub_pred - R[u, i]) ** 2)
                ib_errors.append((ib_pred - R[u, i]) ** 2)
                ub_abs_errors.append(abs(ub_pred - R[u, i]))
                ib_abs_errors.append(abs(ib_pred - R[u, i]))

    ub_rmse = np.sqrt(np.mean(ub_errors)) if ub_errors else 0
    ib_rmse = np.sqrt(np.mean(ib_errors)) if ib_errors else 0
    ub_mae = np.mean(ub_abs_errors) if ub_abs_errors else 0
    ib_mae = np.mean(ib_abs_errors) if ib_abs_errors else 0

    fig, ax = plt.subplots(figsize=(8, 5))
    methods = ['User-CF', 'Item-CF']
    x = np.arange(len(methods))
    width = 0.3
    bars1 = ax.bar(x - width / 2, [ub_rmse, ib_rmse], width, label='RMSE', color='#3498db', alpha=0.85)
    bars2 = ax.bar(x + width / 2, [ub_mae, ib_mae], width, label='MAE', color='#e74c3c', alpha=0.85)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{bar.get_height():.3f}', ha='center', fontsize=10)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{bar.get_height():.3f}', ha='center', fontsize=10)
    ax.set_xlabel('推荐方法')
    ax.set_ylabel('误差')
    ax.set_title('User-CF 与 Item-CF 推荐效果对比')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()
