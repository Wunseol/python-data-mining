"""
模型评估与调优模块 - Model Evaluation & Tuning
================================================
涵盖模型评估与调优的核心技术：
1. 评估指标 (分类/回归/聚类)
2. 交叉验证
3. 网格搜索与随机搜索
4. 学习曲线与验证曲线
5. 模型对比
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold,
    GridSearchCV, RandomizedSearchCV, learning_curve, validation_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 分类评估指标
# ============================================================
def evaluate_classification(y_true, y_pred, y_prob=None):
    """完整的分类评估指标"""
    print("  === 分类评估指标 ===")
    print(f"  准确率 (Accuracy):  {accuracy_score(y_true, y_pred):.4f}")
    print(f"  精确率 (Precision): {precision_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"  召回率 (Recall):    {recall_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"  F1分数:             {f1_score(y_true, y_pred, average='weighted'):.4f}")
    if y_prob is not None:
        print(f"  AUC-ROC:            {roc_auc_score(y_true, y_prob):.4f}")
    print(f"\n  混淆矩阵:\n{confusion_matrix(y_true, y_pred)}")


def plot_confusion_matrix(y_true, y_pred, labels=None, title='混淆矩阵'):
    """可视化混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    n_classes = cm.shape[0]
    tick_labels = labels if labels else range(n_classes)
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel('预测标签')
    ax.set_ylabel('真实标签')
    ax.set_title(title)
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig('混淆矩阵.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_roc_curves(y_true, y_probs_dict):
    """绘制多条ROC曲线"""
    plt.figure(figsize=(8, 6))
    for name, y_prob in y_probs_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC={roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('假正率 (FPR)')
    plt.ylabel('真正率 (TPR)')
    plt.title('ROC曲线对比')
    plt.legend(loc='lower right')
    plt.savefig('ROC曲线对比.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 2. 回归评估指标
# ============================================================
def evaluate_regression(y_true, y_pred):
    """回归评估指标"""
    print("  === 回归评估指标 ===")
    print(f"  MSE:  {mean_squared_error(y_true, y_pred):.4f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
    print(f"  MAE:  {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"  R2:   {r2_score(y_true, y_pred):.4f}")


# ============================================================
# 3. 交叉验证
# ============================================================
def demo_cross_validation(X, y):
    """交叉验证"""
    print("\n--- 交叉验证 ---")
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    }

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"  {'模型':25s} | {'均值':>8s} | {'标准差':>8s} | 各折得分")
    print("  " + "-" * 75)
    for name, model in models.items():
        scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
        print(f"  {name:25s} | {scores.mean():.4f}   | {scores.std():.4f}   | {scores.round(4)}")


# ============================================================
# 4. 网格搜索与随机搜索
# ============================================================
def demo_grid_search(X_train, X_test, y_train, y_test):
    """网格搜索调参"""
    print("\n--- 网格搜索 (GridSearchCV) ---")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto'],
    }
    grid_search = GridSearchCV(SVC(random_state=42), param_grid, cv=5,
                                scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_s, y_train)
    print(f"  最佳参数: {grid_search.best_params_}")
    print(f"  最佳CV分数: {grid_search.best_score_:.4f}")
    print(f"  测试集准确率: {accuracy_score(y_test, grid_search.predict(X_test_s)):.4f}")
    return grid_search


def demo_random_search(X_train, X_test, y_train, y_test):
    """随机搜索调参"""
    print("\n--- 随机搜索 (RandomizedSearchCV) ---")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    from scipy.stats import randint
    param_dist = {
        'n_estimators': randint(50, 200),
        'max_depth': randint(3, 15),
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 5),
    }
    random_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42), param_dist,
        n_iter=20, cv=5, scoring='accuracy', random_state=42, n_jobs=-1
    )
    random_search.fit(X_train_s, y_train)
    print(f"  最佳参数: {random_search.best_params_}")
    print(f"  最佳CV分数: {random_search.best_score_:.4f}")
    print(f"  测试集准确率: {accuracy_score(y_test, random_search.predict(X_test_s)):.4f}")
    return random_search


# ============================================================
# 5. 学习曲线与验证曲线
# ============================================================
def plot_learning_curve_demo(X, y):
    """学习曲线 — 判断过拟合/欠拟合"""
    print("\n--- 学习曲线 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = {
        '逻辑回归': LogisticRegression(max_iter=1000, random_state=42),
        '随机森林(无限制)': RandomForestClassifier(n_estimators=100, random_state=42),
        '随机森林(max_depth=5)': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (name, model) in zip(axes, models.items()):
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_scaled, y, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy'
        )
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)

        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
        ax.plot(train_sizes, train_mean, 'o-', color='blue', label='训练集')
        ax.plot(train_sizes, val_mean, 'o-', color='red', label='验证集')
        ax.set_xlabel('训练样本数')
        ax.set_ylabel('准确率')
        ax.set_title(name)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('学习曲线.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_validation_curve_demo(X, y):
    """验证曲线 — 观察参数对模型性能的影响"""
    print("\n--- 验证曲线 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 随机森林 — n_estimators
    param_range = [10, 30, 50, 80, 100, 150, 200]
    train_scores, val_scores = validation_curve(
        RandomForestClassifier(random_state=42), X_scaled, y,
        param_name='n_estimators', param_range=param_range,
        cv=5, scoring='accuracy', n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(param_range, train_mean, 'o-', color='blue', label='训练集')
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.plot(param_range, val_mean, 'o-', color='red', label='验证集')
    plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    plt.xlabel('n_estimators')
    plt.ylabel('准确率')
    plt.title('验证曲线 — 随机森林 n_estimators')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('验证曲线.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 6. 模型对比
# ============================================================
def compare_models(X, y):
    """多模型综合对比"""
    print("\n--- 模型综合对比 ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
    }

    results = {}
    y_probs = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1': f1_score(y_test, y_pred),
            'AUC': roc_auc_score(y_test, y_prob),
        }
        y_probs[name] = y_prob

    # 打印对比表
    df_results = pd.DataFrame(results).T
    print(df_results.to_string())

    # 雷达图对比
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), subplot_kw=dict(polar=True))
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

    for (name, values), color in zip(results.items(), colors):
        vals = [values[m] for m in metrics] + [values[metrics[0]]]
        ax1.plot(angles, vals, 'o-', linewidth=2, color=color, label=name)
        ax1.fill(angles, vals, alpha=0.05, color=color)

    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(metrics)
    ax1.set_title('模型性能雷达图', pad=20)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1))

    # ROC曲线对比
    for name, y_prob in y_probs.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        ax2.plot(fpr, tpr, lw=2, label=f'{name} (AUC={roc_auc:.3f})')

    ax2.plot([0, 1], [0, 1], 'k--', lw=1)
    ax2.set_xlabel('FPR')
    ax2.set_ylabel('TPR')
    ax2.set_title('ROC曲线对比')
    ax2.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig('模型综合对比.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 主程序演示
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("模型评估与调优完整流程演示")
    print("=" * 60)

    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. 分类评估
    print("\n--- 1. 分类评估指标 ---")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_s, y_train)
    y_pred = rf.predict(X_test_s)
    y_prob = rf.predict_proba(X_test_s)[:, 1]
    evaluate_classification(y_test, y_pred, y_prob)
    plot_confusion_matrix(y_test, y_pred, labels=['恶性', '良性'])

    # 2. 交叉验证
    demo_cross_validation(X, y)

    # 3. 网格搜索
    demo_grid_search(X_train, X_test, y_train, y_test)

    # 4. 随机搜索
    demo_random_search(X_train, X_test, y_train, y_test)

    # 5. 学习曲线
    plot_learning_curve_demo(X, y)

    # 6. 验证曲线
    plot_validation_curve_demo(X, y)

    # 7. 模型综合对比
    compare_models(X, y)
