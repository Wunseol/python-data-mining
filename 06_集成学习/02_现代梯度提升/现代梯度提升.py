"""
现代梯度提升模块 - Modern Gradient Boosting
============================================
涵盖现代梯度提升框架的核心原理与对比实验：
1. LightGBM 核心原理 — 直方图算法、叶子优先生长、GOSS、EFB
2. CatBoost 核心原理 — 有序目标编码、对称决策树、有序提升
3. 三大框架对比实验 — XGBoost vs LightGBM vs CatBoost
4. 可视化对比 — 准确率、训练时间、特征重要性
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from utils import setup_chinese_font

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


# ============================================================
# 1. LightGBM 核心原理讲解
# ============================================================
def explain_lightgbm():
    """LightGBM 核心原理讲解（print-based）"""
    print("=" * 60)
    print("1. LightGBM 核心原理")
    print("=" * 60)

    print("\n--- 1.1 直方图算法 (Histogram-based Algorithm) ---")
    print("  传统方法：对每个特征的每个取值逐一尝试分裂点，时间复杂度 O(#data × #feature)")
    print("  直方图算法：将连续特征离散化为有限个 bin（默认 256 个）")
    print("  步骤：")
    print("    1) 遍历一次数据，统计每个 bin 内的梯度累加和与样本数")
    print("    2) 在 bin 边界上寻找最优分裂点，而非在原始值上搜索")
    print("  优势：")
    print("    - 分裂点搜索空间从 O(#unique_values) 降至 O(#bins)")
    print("    - 直方图做差加速：父节点直方图 - 左子节点 = 右子节点，只需遍历一次")
    print("    - 天然支持稀疏特征，缺失值单独一个 bin")

    print("\n--- 1.2 叶子优先生长 (Leaf-wise Growth) ---")
    print("  层级优先 (Level-wise / XGBoost 默认)：")
    print("    - 按层生长，同层所有节点同时分裂")
    print("    - 优点：不易过拟合，便于并行")
    print("    - 缺点：可能对增益小的叶子做了无意义分裂")
    print("  叶子优先 (Leaf-wise / LightGBM 默认)：")
    print("    - 每次选择当前增益最大的叶子进行分裂")
    print("    - 优点：相同叶子数下损失更低，训练更快")
    print("    - 缺点：可能产生深度不均衡的树，容易过拟合（可通过 max_depth 约束）")

    print("\n--- 1.3 GOSS (Gradient-based One-Side Sampling) ---")
    print("  核心思想：大梯度样本对学习更有价值，小梯度样本可采样")
    print("  步骤：")
    print("    1) 将样本按梯度绝对值排序")
    print("    2) 保留 top-a% 大梯度样本（a 为保留比例）")
    print("    3) 从剩余 (1-a)% 小梯度样本中随机采样 b%")
    print("    4) 对小梯度样本的梯度乘以 (1-a)/b 进行放大，保持数据分布不变")
    print("  效果：在大幅减少样本量的同时，近似保持信息增益的估计精度")

    print("\n--- 1.4 EFB (Exclusive Feature Bundling) ---")
    print("  核心思想：将互斥特征（几乎不同时取非零值）捆绑为一个特征")
    print("  步骤：")
    print("    1) 构建特征冲突图：两个特征同时非零的样本数为冲突数")
    print("    2) 图着色算法：将冲突数低于阈值的特征归为同一颜色（同一 bundle）")
    print("    3) 合并同一 bundle 的特征：偏移原始特征取值使其互不重叠")
    print("  效果：特征数量从 #features 降至 #bundles，加速直方图构建")
    print("  典型场景：独热编码后的稀疏特征天然互斥，EFB 效果显著")


# ============================================================
# 2. CatBoost 核心原理讲解
# ============================================================
def explain_catboost():
    """CatBoost 核心原理讲解（print-based）"""
    print("\n" + "=" * 60)
    print("2. CatBoost 核心原理")
    print("=" * 60)

    print("\n--- 2.1 有序目标编码 (Ordered Target Statistics) ---")
    print("  问题：类别特征直接用目标均值编码会导致目标泄露（Target Leakage）")
    print("  朴素编码：cat_value 的编码 = 该类别下目标变量的均值")
    print("    → 训练时使用了当前样本的标签信息，导致过拟合")
    print("  有序目标编码：")
    print("    1) 引入随机排列 π，将训练样本随机排序")
    print("    2) 对第 i 个样本，仅使用排列中排在 i 之前的样本来计算编码")
    print("    3) 编码公式：target_stats = (count_in_class + prior) / (total_count + prior)")
    print("       其中 count_in_class = 排在 i 之前的同类别样本中正例数")
    print("             total_count = 排在 i 之前的同类别样本数")
    print("             prior = 全局正例比例（平滑项）")
    print("  效果：每个样本的编码仅依赖历史数据，避免目标泄露")

    print("\n--- 2.2 对称决策树 (Oblivious Trees) ---")
    print("  普通决策树：每个节点独立选择最优分裂条件，树结构不对称")
    print("  对称决策树：")
    print("    - 同一层级所有节点使用完全相同的分裂条件和特征")
    print("    - 树的深度固定，叶子节点数为 2^depth")
    print("  优势：")
    print("    - 强正则化效果：限制模型复杂度，防止过拟合")
    print("    - 预测速度快：可用位运算快速索引到叶子节点")
    print("    - 便于模型可视化与解释")

    print("\n--- 2.3 有序提升 (Ordered Boosting) ---")
    print("  问题：标准 Boosting 中，每棵树用所有训练数据计算梯度")
    print("    → 梯度估计使用了当前模型对该样本的预测，产生预测偏移")
    print("  有序提升：")
    print("    1) 维护多个模型状态 M_0, M_1, ..., M_n")
    print("    2) 对第 i 个样本，使用 M_{i-1}（不包含第 i 个样本训练的模型）计算残差")
    print("    3) 每棵树训练时，各样本的梯度由不同的模型状态计算")
    print("  效果：残差估计无偏，消除预测偏移，提升泛化能力")
    print("  实际实现：使用随机排列 + 前缀模型近似，控制计算复杂度")


# ============================================================
# 3. 三大框架对比实验
# ============================================================
def compare_frameworks(X_train, X_test, y_train, y_test, feature_names=None):
    """XGBoost vs LightGBM vs CatBoost 对比实验

    返回:
        results: dict, 包含各框架的准确率、训练时间和模型对象
    """
    print("\n" + "=" * 60)
    print("3. 三大框架对比实验 (XGBoost vs LightGBM vs CatBoost)")
    print("=" * 60)

    results = {}

    if HAS_XGBOOST:
        print("\n--- XGBoost ---")
        xgb = XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            use_label_encoder=False, eval_metric='logloss',
            random_state=42, verbosity=0
        )
        start = time.time()
        xgb.fit(X_train, y_train)
        train_time = time.time() - start
        y_pred = xgb.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"  准确率: {acc:.4f}")
        print(f"  训练时间: {train_time:.4f}s")
        results['XGBoost'] = {
            'model': xgb, 'accuracy': acc, 'time': train_time,
            'importances': xgb.feature_importances_
        }
    else:
        print("\n--- XGBoost 未安装，跳过 ---")
        print("  安装命令: pip install xgboost")

    if HAS_LIGHTGBM:
        print("\n--- LightGBM ---")
        lgbm = lgb.LGBMClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            random_state=42, verbose=-1
        )
        start = time.time()
        lgbm.fit(X_train, y_train)
        train_time = time.time() - start
        y_pred = lgbm.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"  准确率: {acc:.4f}")
        print(f"  训练时间: {train_time:.4f}s")
        results['LightGBM'] = {
            'model': lgbm, 'accuracy': acc, 'time': train_time,
            'importances': lgbm.feature_importances_
        }
    else:
        print("\n--- LightGBM 未安装，跳过 ---")
        print("  安装命令: pip install lightgbm")

    if HAS_CATBOOST:
        print("\n--- CatBoost ---")
        cb = CatBoostClassifier(
            iterations=100, depth=4, learning_rate=0.1,
            random_state=42, verbose=0
        )
        start = time.time()
        cb.fit(X_train, y_train)
        train_time = time.time() - start
        y_pred = cb.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"  准确率: {acc:.4f}")
        print(f"  训练时间: {train_time:.4f}s")
        results['CatBoost'] = {
            'model': cb, 'accuracy': acc, 'time': train_time,
            'importances': cb.feature_importances_
        }
    else:
        print("\n--- CatBoost 未安装，跳过 ---")
        print("  安装命令: pip install catboost")

    if not results:
        print("\n  警告: 所有框架均未安装，无法进行对比实验！")
        print("  请至少安装其中一个: pip install xgboost lightgbm catboost")

    return results


# ============================================================
# 4. 可视化对比
# ============================================================
def plot_comparison(results, feature_names=None):
    """可视化三大框架对比结果

    参数:
        results: dict, compare_frameworks 的返回值
        feature_names: array-like, 特征名称
    """
    if not results:
        print("\n  无对比数据，跳过可视化。")
        return

    names = list(results.keys())
    n_frameworks = len(names)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ---- 准确率对比 ----
    ax1 = axes[0]
    accuracies = [results[n]['accuracy'] for n in names]
    colors = ['#4C72B0', '#55A868', '#C44E52'][:n_frameworks]
    bars = ax1.bar(names, accuracies, color=colors, edgecolor='white', linewidth=1.2)
    ax1.set_ylabel('准确率')
    ax1.set_title('准确率对比')
    ax1.set_ylim(min(accuracies) - 0.05, 1.0)
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{acc:.4f}', ha='center', va='bottom', fontsize=11)

    # ---- 训练时间对比 ----
    ax2 = axes[1]
    times = [results[n]['time'] for n in names]
    bars = ax2.bar(names, times, color=colors, edgecolor='white', linewidth=1.2)
    ax2.set_ylabel('训练时间 (秒)')
    ax2.set_title('训练时间对比')
    for bar, t in zip(bars, times):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                 f'{t:.4f}s', ha='center', va='bottom', fontsize=11)

    # ---- 特征重要性对比 (Top-10) ----
    ax3 = axes[2]
    top_k = 10
    all_importances = []
    for n in names:
        imp = results[n]['importances']
        all_importances.append(imp)

    if feature_names is not None:
        mean_imp = np.mean(all_importances, axis=0)
        top_indices = np.argsort(mean_imp)[::-1][:top_k]
        feat_labels = [feature_names[i] for i in top_indices]
    else:
        top_indices = np.argsort(np.mean(all_importances, axis=0))[::-1][:top_k]
        feat_labels = [f"Feature_{i}" for i in top_indices]

    x_pos = np.arange(top_k)
    bar_width = 0.8 / n_frameworks
    for i, n in enumerate(names):
        imp = results[n]['importances']
        imp_normalized = imp / imp.sum() if imp.sum() > 0 else imp
        offset = (i - n_frameworks / 2 + 0.5) * bar_width
        ax3.barh(x_pos + offset, imp_normalized[top_indices],
                 height=bar_width, label=n, color=colors[i])

    ax3.set_yticks(x_pos)
    ax3.set_yticklabels(feat_labels)
    ax3.invert_yaxis()
    ax3.set_xlabel('归一化重要性')
    ax3.set_title(f'特征重要性 Top-{top_k} 对比')
    ax3.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(_SCRIPT_DIR, '现代梯度提升对比.png'), dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 主程序
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("现代梯度提升完整流程演示")
    print("=" * 60)

    explain_lightgbm()
    explain_catboost()

    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = compare_frameworks(X_train, X_test, y_train, y_test, feature_names)
    plot_comparison(results, feature_names)

    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)
