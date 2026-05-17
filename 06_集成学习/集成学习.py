"""
集成学习模块 - Ensemble Learning
=================================
涵盖集成学习的核心方法：
1. Bagging — 随机森林
2. Boosting — AdaBoost、GBDT、XGBoost
3. Stacking — 堆叠集成
4. Voting — 投票法
5. 集成方法对比
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.ensemble import (
    RandomForestClassifier, BaggingClassifier,
    AdaBoostClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import setup_chinese_font

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# 1. Bagging — 随机森林
# ============================================================
def demo_random_forest(X_train, X_test, y_train, y_test, feature_names=None):
    """随机森林分类 + 特征重要性"""
    print("--- 1. 随机森林 (Bagging) ---")
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"  准确率: {acc:.4f}")

    # 特征重要性
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 5))
    n_features = min(15, len(importances))
    names = feature_names[indices[:n_features]] if feature_names is not None else \
        [f"Feature_{i}" for i in indices[:n_features]]
    plt.bar(range(n_features), importances[indices[:n_features]])
    plt.xticks(range(n_features), names, rotation=45, ha='right')
    plt.title('随机森林特征重要性 Top-15')
    plt.tight_layout()
    plt.savefig(os.path.join(_SCRIPT_DIR, '随机森林特征重要性.png'), dpi=150, bbox_inches='tight')
    plt.show()

    return rf


# ============================================================
# 2. Boosting — AdaBoost
# ============================================================
def demo_adaboost(X_train, X_test, y_train, y_test):
    """AdaBoost分类"""
    print("\n--- 2. AdaBoost (Boosting) ---")
    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=100, learning_rate=0.5, random_state=42
    )
    ada.fit(X_train, y_train)
    y_pred = ada.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"  准确率: {acc:.4f}")

    # 不同弱学习器数量的学习曲线
    train_scores = []
    test_scores = []
    n_estimators_range = range(1, 101)
    for n in n_estimators_range:
        ada_temp = AdaBoostClassifier(n_estimators=n, learning_rate=0.5, random_state=42)
        ada_temp.fit(X_train, y_train)
        train_scores.append(accuracy_score(y_train, ada_temp.predict(X_train)))
        test_scores.append(accuracy_score(y_test, ada_temp.predict(X_test)))

    plt.figure(figsize=(8, 5))
    plt.plot(n_estimators_range, train_scores, label='训练集')
    plt.plot(n_estimators_range, test_scores, label='测试集')
    plt.xlabel('弱学习器数量')
    plt.ylabel('准确率')
    plt.title('AdaBoost学习曲线')
    plt.legend()
    plt.savefig(os.path.join(_SCRIPT_DIR, 'AdaBoost学习曲线.png'), dpi=150, bbox_inches='tight')
    plt.show()

    return ada


# ============================================================
# 3. Boosting — GBDT
# ============================================================
def demo_gbdt(X_train, X_test, y_train, y_test):
    """GBDT梯度提升树"""
    print("\n--- 3. GBDT (Gradient Boosting) ---")
    gbdt = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        subsample=0.8, random_state=42
    )
    gbdt.fit(X_train, y_train)
    y_pred = gbdt.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"  准确率: {acc:.4f}")

    # 训练过程deviance
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(gbdt.train_score_) + 1), gbdt.train_score_, label='训练损失')
    plt.xlabel('迭代次数')
    plt.ylabel('训练损失 (Deviance)')
    plt.title('GBDT训练过程')
    plt.legend()
    plt.savefig(os.path.join(_SCRIPT_DIR, 'GBDT训练过程.png'), dpi=150, bbox_inches='tight')
    plt.show()

    return gbdt


# ============================================================
# 4. Voting — 投票法
# ============================================================
def demo_voting(X_train, X_test, y_train, y_test):
    """软/硬投票集成"""
    print("\n--- 4. Voting (投票法) ---")
    estimators = [
        ('lr', LogisticRegression(max_iter=1000, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('svc', SVC(probability=True, random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=5)),
    ]

    # 硬投票
    voting_hard = VotingClassifier(estimators=estimators, voting='hard')
    voting_hard.fit(X_train, y_train)
    acc_hard = accuracy_score(y_test, voting_hard.predict(X_test))

    # 软投票
    voting_soft = VotingClassifier(estimators=estimators, voting='soft')
    voting_soft.fit(X_train, y_train)
    acc_soft = accuracy_score(y_test, voting_soft.predict(X_test))

    print(f"  硬投票准确率: {acc_hard:.4f}")
    print(f"  软投票准确率: {acc_soft:.4f}")

    # 对比单个基学习器
    print("  单个基学习器:")
    for name, model in estimators:
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"    {name}: {acc:.4f}")

    return voting_soft


# ============================================================
# 5. Stacking — 堆叠集成
# ============================================================
def demo_stacking(X_train, X_test, y_train, y_test):
    """Stacking堆叠集成"""
    print("\n--- 5. Stacking (堆叠) ---")
    base_estimators = [
        ('lr', LogisticRegression(max_iter=1000, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=5)),
        ('dt', DecisionTreeClassifier(max_depth=5, random_state=42)),
    ]
    meta_estimator = LogisticRegression(max_iter=1000, random_state=42)

    stacking = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_estimator,
        cv=5
    )
    stacking.fit(X_train, y_train)
    y_pred = stacking.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"  Stacking准确率: {acc:.4f}")

    # 交叉验证对比
    print("  交叉验证对比:")
    cv_scores = cross_val_score(stacking, X_train, y_train, cv=5, scoring='accuracy')
    print(f"    Stacking CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    for name, model in base_estimators:
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        print(f"    {name} CV: {scores.mean():.4f} ± {scores.std():.4f}")

    return stacking


# ============================================================
# 6. XGBoost (如果安装)
# ============================================================
def demo_xgboost(X_train, X_test, y_train, y_test):
    """XGBoost分类"""
    print("\n--- 6. XGBoost ---")
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                            use_label_encoder=False, eval_metric='logloss',
                            random_state=42)
        xgb.fit(X_train, y_train)
        y_pred = xgb.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"  XGBoost准确率: {acc:.4f}")
        return xgb
    except ImportError:
        print("  XGBoost未安装，请运行: pip install xgboost")
        return None


# ============================================================
# 7. Bagging 手动实现
# ============================================================
class BaggingManual:
    """Bagging（装袋法）手动实现

    核心思想：
    1. 对训练集进行有放回抽样（Bootstrap），生成多个子训练集
    2. 在每个子训练集上训练一个基学习器（决策树桩）
    3. 对所有基学习器的预测结果进行多数投票
    """

    def __init__(self, n_estimators=10, max_samples=1.0, random_state=42):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.estimators_ = []

    def fit(self, X, y):
        """使用 Bootstrap 采样训练多个基学习器

        参数:
            X: 训练特征, shape=(n_samples, n_features)
            y: 训练标签, shape=(n_samples,)
        """
        rng = np.random.RandomState(self.random_state)
        n_samples = X.shape[0]
        # 每个子训练集的样本数
        sub_sample_size = int(n_samples * self.max_samples)

        self.estimators_ = []
        for i in range(self.n_estimators):
            # Bootstrap 有放回抽样：从 n_samples 中抽取 sub_sample_size 个样本
            indices = rng.choice(n_samples, size=sub_sample_size, replace=True)
            X_sub, y_sub = X[indices], y[indices]

            # 训练决策树桩作为基学习器（max_depth=1）
            stump = DecisionTreeClassifier(max_depth=1, random_state=rng.randint(0, 2**31))
            stump.fit(X_sub, y_sub)
            self.estimators_.append(stump)

        return self

    def predict(self, X):
        """多数投票预测

        对每个样本，统计所有基学习器的预测结果，取出现次数最多的类别
        """
        # 收集所有基学习器的预测: shape=(n_estimators, n_samples)
        predictions = np.array([est.predict(X) for est in self.estimators_])
        # 对每列（每个样本）取众数
        result = np.array([
            np.bincount(predictions[:, i].astype(int)).argmax()
            for i in range(X.shape[0])
        ])
        return result


# ============================================================
# 8. AdaBoost 手动实现
# ============================================================
class AdaBoostManual:
    """AdaBoost（自适应提升）手动实现

    核心思想：
    1. 初始化每个样本的权重为 1/N
    2. 每轮训练一个弱学习器（决策树桩）
    3. 根据弱学习器的加权错误率计算该学习器的权重 α
    4. 增大被错误分类样本的权重，减小被正确分类样本的权重
    5. 最终通过加权投票进行预测

    α = learning_rate * ln((1 - err) / err)
    权重更新: w_i *= exp(α * I(y_i ≠ h(x_i)))
    """

    def __init__(self, n_estimators=50, learning_rate=1.0, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.estimators_ = []
        self.estimator_weights_ = []

    def fit(self, X, y):
        """训练 AdaBoost 集成模型

        参数:
            X: 训练特征, shape=(n_samples, n_features)
            y: 训练标签, shape=(n_samples,)，需为 {0, 1} 二分类
        """
        rng = np.random.RandomState(self.random_state)
        n_samples = X.shape[0]

        # 步骤1: 初始化样本权重，每个样本权重相等
        sample_weights = np.ones(n_samples) / n_samples

        self.estimators_ = []
        self.estimator_weights_ = []

        for t in range(self.n_estimators):
            # 步骤2: 根据当前样本权重训练弱学习器（决策树桩）
            stump = DecisionTreeClassifier(max_depth=1, random_state=rng.randint(0, 2**31))
            stump.fit(X, y, sample_weight=sample_weights)

            # 步骤3: 计算加权错误率
            y_pred = stump.predict(X)
            incorrect = (y_pred != y)
            err = np.dot(sample_weights, incorrect)

            # 如果错误率 >= 0.5，说明该弱学习器比随机猜测还差，提前终止
            if err >= 0.5:
                break

            # 避免除零：err 过小时截断
            err = max(err, 1e-10)

            # 步骤4: 计算弱学习器权重 α
            # α 越大表示该弱学习器越重要
            alpha = self.learning_rate * np.log((1.0 - err) / err)

            # 步骤5: 更新样本权重
            # 被错误分类的样本权重增大，被正确分类的样本权重减小
            sample_weights *= np.exp(alpha * incorrect)

            # 归一化样本权重，使其总和为1
            sample_weights /= sample_weights.sum()

            self.estimators_.append(stump)
            self.estimator_weights_.append(alpha)

        self.estimator_weights_ = np.array(self.estimator_weights_)
        return self

    def predict(self, X):
        """加权投票预测

        每个弱学习器的预测乘以其权重 α，求和后取符号
        """
        # 每个弱学习器的预测: shape=(n_estimators, n_samples)
        predictions = np.array([est.predict(X) for est in self.estimators_])

        # 将标签 {0, 1} 映射到 {-1, +1} 以便加权求和
        predictions_signed = 2 * predictions - 1

        # 加权求和: shape=(n_samples,)
        weighted_sum = np.dot(self.estimator_weights_, predictions_signed)

        # 取符号，再映射回 {0, 1}
        result = (np.sign(weighted_sum) >= 0).astype(int)
        return result


# ============================================================
# 主程序演示
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("集成学习完整流程演示")
    print("=" * 60)

    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    rf = demo_random_forest(X_train_s, X_test_s, y_train, y_test, feature_names)
    ada = demo_adaboost(X_train_s, X_test_s, y_train, y_test)
    gbdt = demo_gbdt(X_train_s, X_test_s, y_train, y_test)
    voting = demo_voting(X_train_s, X_test_s, y_train, y_test)
    stacking = demo_stacking(X_train_s, X_test_s, y_train, y_test)
    xgb = demo_xgboost(X_train_s, X_test_s, y_train, y_test)

    # ============================================================
    # 手动实现 vs sklearn 对比
    # ============================================================
    print("\n" + "=" * 60)
    print("手动实现 vs sklearn 对比")
    print("=" * 60)

    # --- Bagging 对比 ---
    print("\n--- Bagging 手动实现 vs sklearn ---")
    bag_manual = BaggingManual(n_estimators=10, max_samples=1.0, random_state=42)
    bag_manual.fit(X_train_s, y_train)
    y_pred_bag_manual = bag_manual.predict(X_test_s)
    acc_bag_manual = accuracy_score(y_test, y_pred_bag_manual)

    bag_sklearn = BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=10, max_samples=1.0, random_state=42
    )
    bag_sklearn.fit(X_train_s, y_train)
    y_pred_bag_sklearn = bag_sklearn.predict(X_test_s)
    acc_bag_sklearn = accuracy_score(y_test, y_pred_bag_sklearn)

    print(f"  BaggingManual 准确率:   {acc_bag_manual:.4f}")
    print(f"  sklearn Bagging 准确率: {acc_bag_sklearn:.4f}")

    # --- AdaBoost 对比 ---
    print("\n--- AdaBoost 手动实现 vs sklearn ---")
    ada_manual = AdaBoostManual(n_estimators=50, learning_rate=1.0, random_state=42)
    ada_manual.fit(X_train_s, y_train)
    y_pred_ada_manual = ada_manual.predict(X_test_s)
    acc_ada_manual = accuracy_score(y_test, y_pred_ada_manual)

    ada_sklearn = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=50, learning_rate=1.0, random_state=42
    )
    ada_sklearn.fit(X_train_s, y_train)
    y_pred_ada_sklearn = ada_sklearn.predict(X_test_s)
    acc_ada_sklearn = accuracy_score(y_test, y_pred_ada_sklearn)

    print(f"  AdaBoostManual 准确率:   {acc_ada_manual:.4f}")
    print(f"  sklearn AdaBoost 准确率: {acc_ada_sklearn:.4f}")
