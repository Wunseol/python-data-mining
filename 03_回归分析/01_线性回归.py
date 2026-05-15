"""
线性回归模块 - Linear Regression
=================================
涵盖数据挖掘中线性回归的核心内容：
1. 简单线性回归 (一元)
2. 多元线性回归
3. 正则化回归 (Ridge、Lasso、ElasticNet)
4. 梯度下降实现
5. 回归诊断与评估
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression, load_diabetes
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 简单线性回归 (手动实现)
# ============================================================
class SimpleLinearRegression:
    """一元线性回归：y = wx + b，最小二乘法"""

    def fit(self, X, y):
        n = len(X)
        self.w = (n * np.sum(X * y) - np.sum(X) * np.sum(y)) / \
                 (n * np.sum(X ** 2) - np.sum(X) ** 2)
        self.b = np.mean(y) - self.w * np.mean(X)
        return self

    def predict(self, X):
        return self.w * X + self.b

    def r2(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot


# ============================================================
# 2. 多元线性回归 (梯度下降实现)
# ============================================================
class LinearRegressionGD:
    """多元线性回归 — 梯度下降"""

    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.losses = []

    def fit(self, X, y):
        m, n = X.shape
        self.W = np.zeros(n)
        self.b = 0

        for i in range(self.n_iters):
            y_pred = X @ self.W + self.b
            error = y_pred - y
            dW = (2 / m) * (X.T @ error)
            db = (2 / m) * np.sum(error)
            self.W -= self.lr * dW
            self.b -= self.lr * db

            loss = np.mean(error ** 2)
            self.losses.append(loss)

        return self

    def predict(self, X):
        return X @ self.W + self.b


# ============================================================
# 3. 正则化回归对比
# ============================================================
def compare_regularization(X_train, X_test, y_train, y_test):
    """对比Ridge、Lasso、ElasticNet"""
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge (α=1)': Ridge(alpha=1),
        'Ridge (α=10)': Ridge(alpha=10),
        'Lasso (α=1)': Lasso(alpha=1, max_iter=10000),
        'Lasso (α=0.1)': Lasso(alpha=0.1, max_iter=10000),
        'ElasticNet (α=1)': ElasticNet(alpha=1, l1_ratio=0.5, max_iter=10000),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        results[name] = {'RMSE': rmse, 'R2': r2}
        print(f"  {name:25s} | RMSE={rmse:.2f} | R2={r2:.4f}")

    return results


# ============================================================
# 4. 回归诊断
# ============================================================
def regression_diagnostics(model, X_test, y_test, y_pred):
    """回归诊断：残差分析"""
    residuals = y_test - y_pred

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 残差 vs 预测值
    axes[0].scatter(y_pred, residuals, alpha=0.5, s=10)
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].set_xlabel('预测值')
    axes[0].set_ylabel('残差')
    axes[0].set_title('残差 vs 预测值')

    # 残差直方图
    axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('残差')
    axes[1].set_title('残差分布')

    # Q-Q图
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[2])
    axes[2].set_title('Q-Q图')

    plt.tight_layout()
    plt.savefig('回归诊断图.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("回归诊断图已保存")


# ============================================================
# 5. 可视化
# ============================================================
def plot_simple_regression():
    """可视化简单线性回归"""
    np.random.seed(42)
    X = np.random.uniform(0, 10, 100)
    y = 2.5 * X + 3 + np.random.normal(0, 2, 100)

    model = SimpleLinearRegression()
    model.fit(X, y)

    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, alpha=0.5, label='数据点')
    plt.plot(X, model.predict(X), 'r-', linewidth=2,
             label=f'拟合线: y={model.w:.2f}x+{model.b:.2f}')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('简单线性回归')
    plt.legend()
    plt.savefig('简单线性回归.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"斜率 w={model.w:.4f}, 截距 b={model.b:.4f}, R²={model.r2(X, y):.4f}")


def plot_gradient_descent_loss():
    """可视化梯度下降损失曲线"""
    X, y = make_regression(n_samples=200, n_features=5, noise=10, random_state=42)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    model = LinearRegressionGD(lr=0.1, n_iters=200)
    model.fit(X, y)

    plt.figure(figsize=(8, 5))
    plt.plot(range(len(model.losses)), model.losses)
    plt.xlabel('迭代次数')
    plt.ylabel('MSE损失')
    plt.title('梯度下降损失收敛曲线')
    plt.savefig('梯度下降损失曲线.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 主程序演示
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("线性回归完整流程演示")
    print("=" * 60)

    # --- 简单线性回归 ---
    print("\n--- 1. 简单线性回归 ---")
    plot_simple_regression()

    # --- 梯度下降实现 ---
    print("\n--- 2. 梯度下降实现 ---")
    plot_gradient_descent_loss()

    # --- 多元线性回归 + 正则化 ---
    print("\n--- 3. 糖尿病数据集 — 正则化回归对比 ---")
    data = load_diabetes()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = compare_regularization(X_train_s, X_test_s, y_train, y_test)

    # --- 回归诊断 ---
    print("\n--- 4. 回归诊断 ---")
    lr = LinearRegression()
    lr.fit(X_train_s, y_train)
    y_pred = lr.predict(X_test_s)
    print(f"  RMSE={np.sqrt(mean_squared_error(y_test, y_pred)):.2f}, R2={r2_score(y_test, y_pred):.4f}")
    regression_diagnostics(lr, X_test_s, y_test, y_pred)
