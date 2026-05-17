"""
可解释AI模块 - Explainable AI (XAI)
====================================
涵盖可解释人工智能的核心技术：
1. 可解释性概述（内在可解释 vs 事后解释、全局 vs 局部）
2. LIME 手动实现（局部线性近似解释）
3. SHAP 值手动实现（简化 KernelSHAP）
4. 部分依赖图(PDP)与个体条件期望图(ICE)
5. 可视化（LIME柱状图、SHAP摘要图/瀑布图、PDP/ICE曲线）
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from utils import setup_chinese_font

setup_chinese_font()


# ============================================================
# 1. 可解释性概述
# ============================================================
def print_xai_overview():
    """打印可解释AI概述信息"""
    print("=" * 60)
    print("可解释AI概述")
    print("=" * 60)

    print("\n--- 内在可解释 vs 事后解释 ---")
    print("  内在可解释 (Intrinsic): 模型本身可解释")
    print("    - 线性回归: 系数直接反映特征重要性")
    print("    - 决策树: 分裂路径直观可读")
    print("    - 朴素贝叶斯: 概率贡献可分解")
    print("    - 广义加性模型(GAM): 各特征加性贡献")
    print()
    print("  事后解释 (Post-hoc): 对已训练模型进行解释")
    print("    - LIME: 局部线性近似")
    print("    - SHAP: 基于博弈论的Shapley值")
    print("    - PDP/ICE: 特征边际效应")
    print("    - 特征重要性: 基于置换或基尼重要性")

    print("\n--- 全局解释 vs 局部解释 ---")
    print("  全局解释 (Global): 理解模型整体行为")
    print("    - 特征重要性排序")
    print("    - 部分依赖图(PDP)")
    print("    - 全局代理模型")
    print()
    print("  局部解释 (Local): 理解单个预测的原因")
    print("    - LIME局部近似")
    print("    - SHAP个体解释")
    print("    - ICE个体曲线")
    print("    - 反事实解释")

    print("\n--- 常见可解释方法对比 ---")
    comparison_data = {
        '方法': ['LIME', 'SHAP', 'PDP', 'ICE', '特征重要性', '决策树'],
        '类型': ['事后', '事后', '事后', '事后', '事后', '内在'],
        '范围': ['局部', '局部+全局', '全局', '局部', '全局', '局部+全局'],
        '模型无关': ['是', '是', '是', '是', '否(部分)', '否'],
        '计算开销': ['低', '高', '中', '中', '低', '无'],
    }
    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False))


# ============================================================
# 2. LIME 手动实现
# ============================================================
class LIMEManual:
    """LIME (Local Interpretable Model-agnostic Explanations) 手动实现

    通过在待解释实例附近生成扰动样本，用局部线性模型近似黑箱模型。
    """

    def __init__(self, model, X_train):
        self.model = model
        self.X_train = np.array(X_train)
        self.feature_means = np.mean(self.X_train, axis=0)
        self.feature_stds = np.std(self.X_train, axis=0)
        self.feature_stds[self.feature_stds == 0] = 1.0

    def _generate_perturbed_samples(self, instance, n_samples):
        perturbed = np.random.normal(
            loc=self.feature_means,
            scale=self.feature_stds,
            size=(n_samples, len(self.feature_means))
        )
        return perturbed

    def _compute_weights(self, perturbed, instance, kernel_width):
        distances = np.sqrt(np.sum((perturbed - instance) ** 2, axis=1))
        weights = np.exp(-(distances ** 2) / (kernel_width ** 2))
        return weights

    def explain(self, instance, n_samples=5000, kernel_width=1.0):
        instance = np.array(instance).flatten()
        n_features = len(instance)

        perturbed = self._generate_perturbed_samples(instance, n_samples)
        perturbed_predictions = self.model.predict_proba(perturbed)[:, 1]

        weights = self._compute_weights(perturbed, instance, kernel_width)

        perturbed_normalized = (perturbed - self.feature_means) / self.feature_stds
        instance_normalized = (instance - self.feature_means) / self.feature_stds

        X_local = np.column_stack([perturbed_normalized, np.ones(n_samples)])
        y_local = perturbed_predictions

        W = np.diag(weights)
        try:
            beta = np.linalg.inv(X_local.T @ W @ X_local) @ X_local.T @ W @ y_local
        except np.linalg.LinAlgError:
            ridge = Ridge(alpha=1.0)
            ridge.fit(perturbed_normalized, y_local, sample_weight=weights)
            beta = np.concatenate([ridge.coef_, [ridge.intercept_]])

        feature_contributions = beta[:n_features]
        intercept = beta[n_features]

        return {
            'feature_contributions': feature_contributions,
            'intercept': intercept,
            'perturbed_samples': perturbed,
            'perturbed_predictions': perturbed_predictions,
            'weights': weights,
        }


# ============================================================
# 3. SHAP 值手动实现
# ============================================================
class SHAPManual:
    """简化 KernelSHAP 手动实现

    通过采样近似计算 Shapley 值，衡量每个特征对预测的边际贡献。
    """

    def __init__(self, model, X_train, n_samples=1000):
        self.model = model
        self.X_train = np.array(X_train)
        self.n_background = min(50, len(self.X_train))
        rng = np.random.RandomState(42)
        self.background = self.X_train[rng.choice(len(self.X_train), self.n_background, replace=False)]
        self.n_samples = n_samples

    def _predict_with_coalition(self, instance, coalition, background):
        n_features = len(instance)
        masked = background.copy()
        for i in coalition:
            masked[:, i] = instance[i]
        predictions = self.model.predict_proba(masked)[:, 1]
        return np.mean(predictions)

    def explain(self, instance):
        instance = np.array(instance).flatten()
        n_features = len(instance)
        shap_values = np.zeros(n_features)

        all_features = list(range(n_features))

        for _ in range(self.n_samples):
            perm = np.random.permutation(n_features)
            prev_pred = self._predict_with_coalition(instance, [], self.background)

            for i in range(n_features):
                feature_idx = perm[i]
                coalition = list(perm[:i + 1])
                current_pred = self._predict_with_coalition(instance, coalition, self.background)
                shap_values[feature_idx] += (current_pred - prev_pred)
                prev_pred = current_pred

        shap_values /= self.n_samples
        base_value = self._predict_with_coalition(instance, [], self.background)

        return {
            'shap_values': shap_values,
            'base_value': base_value,
        }


# ============================================================
# 4. 部分依赖图(PDP)与个体条件期望图(ICE)
# ============================================================
def compute_pdp(model, X, feature_idx, grid_points=50):
    """计算部分依赖图(PDP)值

    对指定特征在网格点上取值，其余特征保持原值，
    对所有样本预测取平均，得到该特征对预测的边际效应。
    """
    X = np.array(X)
    grid = np.linspace(X[:, feature_idx].min(), X[:, feature_idx].max(), grid_points)
    pdp_values = np.zeros(grid_points)

    for i, val in enumerate(grid):
        X_temp = X.copy()
        X_temp[:, feature_idx] = val
        predictions = model.predict_proba(X_temp)[:, 1]
        pdp_values[i] = np.mean(predictions)

    return grid, pdp_values


def compute_ice(model, X, feature_idx, grid_points=50):
    """计算个体条件期望图(ICE)值

    对每个样本单独计算指定特征变化时的预测曲线，
    反映个体层面的特征效应异质性。
    """
    X = np.array(X)
    grid = np.linspace(X[:, feature_idx].min(), X[:, feature_idx].max(), grid_points)
    n_samples = X.shape[0]
    ice_values = np.zeros((n_samples, grid_points))

    for i, val in enumerate(grid):
        X_temp = X.copy()
        X_temp[:, feature_idx] = val
        predictions = model.predict_proba(X_temp)[:, 1]
        ice_values[:, i] = predictions

    return grid, ice_values


# ============================================================
# 5. 可视化
# ============================================================
def plot_lime_explanation(explanation, feature_names, title='LIME局部解释'):
    """绘制LIME局部解释柱状图"""
    contributions = explanation['feature_contributions']
    sorted_idx = np.argsort(np.abs(contributions))[::-1]
    top_n = min(10, len(sorted_idx))
    top_idx = sorted_idx[:top_n]

    top_contributions = contributions[top_idx]
    top_features = [feature_names[i] for i in top_idx]

    colors = ['#ff6b6b' if c > 0 else '#4ecdc4' for c in top_contributions]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(top_n)
    ax.barh(y_pos, top_contributions, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features)
    ax.set_xlabel('特征贡献值')
    ax.set_title(title)
    ax.invert_yaxis()
    ax.axvline(x=0, color='black', linewidth=0.8)
    plt.tight_layout()
    plt.savefig('LIME局部解释.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_shap_summary(shap_values_list, feature_names, title='SHAP摘要图'):
    """绘制SHAP摘要图（点图）"""
    shap_matrix = np.array(shap_values_list)
    n_features = shap_matrix.shape[1]
    mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
    sorted_idx = np.argsort(mean_abs_shap)[::-1]
    top_n = min(10, len(sorted_idx))
    top_idx = sorted_idx[:top_n]

    fig, ax = plt.subplots(figsize=(10, 6))
    for rank, idx in enumerate(top_idx):
        vals = shap_matrix[:, idx]
        jitter = np.random.RandomState(42).uniform(-0.3, 0.3, size=len(vals))
        ax.scatter(vals, np.full(len(vals), rank) + jitter,
                   alpha=0.4, s=10, c=vals, cmap='coolwarm',
                   vmin=-np.max(np.abs(shap_matrix)),
                   vmax=np.max(np.abs(shap_matrix)))

    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in top_idx])
    ax.set_xlabel('SHAP值')
    ax.set_title(title)
    ax.invert_yaxis()
    plt.colorbar(ax.collections[0], ax=ax, label='SHAP值')
    plt.tight_layout()
    plt.savefig('SHAP摘要图.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_shap_waterfall(shap_result, feature_names, title='SHAP瀑布图'):
    """绘制SHAP瀑布图（单个实例）"""
    shap_values = shap_result['shap_values']
    base_value = shap_result['base_value']
    sorted_idx = np.argsort(np.abs(shap_values))[::-1]
    top_n = min(10, len(sorted_idx))
    top_idx = sorted_idx[:top_n]

    top_shap = shap_values[top_idx]
    top_features = [feature_names[i] for i in top_idx]

    cumulative = np.zeros(top_n + 1)
    cumulative[0] = base_value
    for i in range(top_n):
        cumulative[i + 1] = cumulative[i] + top_shap[i]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(top_n + 1)
    labels = ['基准值'] + [f'{f} ({s:+.4f})' for f, s in zip(top_features, top_shap)]

    bar_values = np.zeros(top_n + 1)
    bar_values[0] = base_value
    for i in range(top_n):
        bar_values[i + 1] = top_shap[i]

    colors = ['#4ecdc4' if v >= 0 else '#ff6b6b' for v in bar_values]
    colors[0] = '#95a5a6'

    ax.barh(y_pos, bar_values, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('SHAP值 / 贡献')
    ax.set_title(title)
    ax.invert_yaxis()
    ax.axvline(x=0, color='black', linewidth=0.8)
    plt.tight_layout()
    plt.savefig('SHAP瀑布图.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_pdp_ice(grid, pdp_values, ice_values, feature_name, title='PDP/ICE曲线'):
    """绘制PDP和ICE曲线"""
    fig, ax = plt.subplots(figsize=(10, 6))

    n_show = min(50, ice_values.shape[0])
    for i in range(n_show):
        ax.plot(grid, ice_values[i], color='lightblue', alpha=0.3, linewidth=0.5)

    ax.plot(grid, pdp_values, color='red', linewidth=2.5, label='PDP (平均)')

    ax.set_xlabel(feature_name)
    ax.set_ylabel('预测概率')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig('PDP_ICE曲线.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 主程序
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("可解释AI (XAI) 实践")
    print("=" * 60)

    # ---- 1. 可解释性概述 ----
    print_xai_overview()

    # ---- 数据准备 ----
    print("\n" + "=" * 60)
    print("数据准备与模型训练")
    print("=" * 60)

    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)

    train_acc = rf.score(X_train_scaled, y_train)
    test_acc = rf.score(X_test_scaled, y_test)
    print(f"  随机森林训练准确率: {train_acc:.4f}")
    print(f"  随机森林测试准确率: {test_acc:.4f}")

    explain_idx = 0
    instance = X_test_scaled[explain_idx]
    true_label = y_test[explain_idx]
    pred_label = rf.predict(instance.reshape(1, -1))[0]
    pred_prob = rf.predict_proba(instance.reshape(1, -1))[0, 1]
    print(f"\n  待解释样本索引: {explain_idx}")
    print(f"  真实标签: {true_label}, 预测标签: {pred_label}, 预测概率: {pred_prob:.4f}")

    # ---- 2. LIME 手动实现 ----
    print("\n" + "=" * 60)
    print("LIME 手动实现")
    print("=" * 60)

    lime = LIMEManual(rf, X_train_scaled)
    lime_result = lime.explain(instance, n_samples=5000, kernel_width=1.0)

    print(f"  截距: {lime_result['intercept']:.4f}")
    print(f"  Top-5 特征贡献:")
    contributions = lime_result['feature_contributions']
    sorted_idx = np.argsort(np.abs(contributions))[::-1]
    for rank, idx in enumerate(sorted_idx[:5]):
        print(f"    {rank + 1}. {feature_names[idx]}: {contributions[idx]:+.4f}")

    plot_lime_explanation(lime_result, feature_names)

    # ---- 3. SHAP 值手动实现 ----
    print("\n" + "=" * 60)
    print("SHAP 值手动实现")
    print("=" * 60)

    shap_explainer = SHAPManual(rf, X_train_scaled, n_samples=500)
    shap_result = shap_explainer.explain(instance)

    print(f"  基准值 (Base Value): {shap_result['base_value']:.4f}")
    print(f"  Top-5 SHAP值:")
    shap_vals = shap_result['shap_values']
    sorted_shap_idx = np.argsort(np.abs(shap_vals))[::-1]
    for rank, idx in enumerate(sorted_shap_idx[:5]):
        print(f"    {rank + 1}. {feature_names[idx]}: {shap_vals[idx]:+.4f}")
    print(f"  基准值 + SHAP值之和: {shap_result['base_value'] + np.sum(shap_vals):.4f}")
    print(f"  模型预测概率: {pred_prob:.4f}")

    n_explain = 20
    print(f"\n  计算 {n_explain} 个样本的SHAP值用于摘要图...")
    all_shap_values = []
    for i in range(n_explain):
        result = shap_explainer.explain(X_test_scaled[i])
        all_shap_values.append(result['shap_values'])

    plot_shap_summary(all_shap_values, feature_names)
    plot_shap_waterfall(shap_result, feature_names)

    # ---- 4. PDP/ICE ----
    print("\n" + "=" * 60)
    print("部分依赖图(PDP)与个体条件期望图(ICE)")
    print("=" * 60)

    top_feature_idx = sorted_shap_idx[0]
    top_feature_name = feature_names[top_feature_idx]
    print(f"  选择最重要特征: {top_feature_name} (索引 {top_feature_idx})")

    grid, pdp_vals = compute_pdp(rf, X_test_scaled, top_feature_idx, grid_points=30)
    print(f"  PDP值范围: [{pdp_vals.min():.4f}, {pdp_vals.max():.4f}]")

    grid, ice_vals = compute_ice(rf, X_test_scaled, top_feature_idx, grid_points=30)
    print(f"  ICE曲线数量: {ice_vals.shape[0]}")

    plot_pdp_ice(grid, pdp_vals, ice_vals, top_feature_name,
                 title=f'PDP/ICE - {top_feature_name}')

    print("\n" + "=" * 60)
    print("可解释AI实践完成")
    print("=" * 60)
