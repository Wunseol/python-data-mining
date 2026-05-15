"""
时间序列分析模块 - Time Series Analysis
========================================
涵盖时间序列分析的核心内容：
1. 时间序列基础 (平稳性检验、自相关)
2. ARIMA模型
3. 指数平滑法
4. 时间序列分解
5. 趋势与季节性分析
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 辅助函数：生成模拟时间序列
# ============================================================
def create_time_series(n_days=365 * 3, seed=42):
    """生成含趋势、季节性、噪声的模拟时间序列"""
    np.random.seed(seed)
    dates = pd.date_range(start='2021-01-01', periods=n_days, freq='D')

    # 趋势
    trend = np.linspace(50, 120, n_days)
    # 季节性 (年度周期)
    seasonality = 15 * np.sin(2 * np.pi * np.arange(n_days) / 365)
    # 周效应
    weekly = 3 * np.sin(2 * np.pi * np.arange(n_days) / 7)
    # 噪声
    noise = np.random.normal(0, 5, n_days)

    values = trend + seasonality + weekly + noise
    ts = pd.Series(values, index=dates, name='value')
    return ts


# ============================================================
# 1. 时间序列基础分析
# ============================================================
def basic_analysis(ts):
    """时间序列基础分析：平稳性检验、自相关"""
    print("--- 时间序列基础分析 ---")
    print(f"  时间范围: {ts.index[0].date()} ~ {ts.index[-1].date()}")
    print(f"  样本数: {len(ts)}")
    print(f"  均值: {ts.mean():.2f}, 标准差: {ts.std():.2f}")

    # ADF检验 (平稳性)
    try:
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(ts.dropna())
        print(f"\n  ADF检验:")
        print(f"    ADF统计量: {result[0]:.4f}")
        print(f"    p值: {result[1]:.4f}")
        print(f"    {'平稳' if result[1] < 0.05 else '非平稳'} (5%显著性水平)")
    except ImportError:
        print("  [提示] 需安装statsmodels: pip install statsmodels")

    # 绘制时间序列 + 滚动统计
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 原始序列
    axes[0, 0].plot(ts, linewidth=0.8)
    axes[0, 0].set_title('原始时间序列')
    axes[0, 0].set_ylabel('值')

    # 滚动均值与标准差
    rolling_mean = ts.rolling(window=30).mean()
    rolling_std = ts.rolling(window=30).std()
    axes[0, 1].plot(ts, alpha=0.3, linewidth=0.5)
    axes[0, 1].plot(rolling_mean, color='red', label='30日滚动均值')
    axes[0, 1].plot(rolling_std, color='green', label='30日滚动标准差')
    axes[0, 1].set_title('滚动统计量')
    axes[0, 1].legend()

    # ACF图
    try:
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        plot_acf(ts.dropna(), lags=50, ax=axes[1, 0])
        axes[1, 0].set_title('自相关函数 (ACF)')
        plot_pacf(ts.dropna(), lags=50, ax=axes[1, 1])
        axes[1, 1].set_title('偏自相关函数 (PACF)')
    except ImportError:
        axes[1, 0].text(0.5, 0.5, '需要statsmodels', ha='center', va='center')
        axes[1, 1].text(0.5, 0.5, '需要statsmodels', ha='center', va='center')

    plt.tight_layout()
    plt.savefig('时间序列基础分析.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 2. 时间序列分解
# ============================================================
def decompose_ts(ts, model='additive', period=365):
    """STL时间序列分解：趋势 + 季节 + 残差"""
    print("\n--- 时间序列分解 ---")
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        result = seasonal_decompose(ts, model=model, period=period)

        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        result.observed.plot(ax=axes[0], title='原始序列')
        result.trend.plot(ax=axes[1], title='趋势')
        result.seasonal.plot(ax=axes[2], title='季节性')
        result.resid.plot(ax=axes[3], title='残差')
        plt.tight_layout()
        plt.savefig('时间序列分解.png', dpi=150, bbox_inches='tight')
        plt.show()
        return result
    except ImportError:
        print("  [提示] 需安装statsmodels: pip install statsmodels")
        return None


# ============================================================
# 3. ARIMA模型
# ============================================================
def demo_arima(ts, order=(1, 1, 1), forecast_steps=30):
    """ARIMA模型拟合与预测"""
    print("\n--- ARIMA模型 ---")
    try:
        from statsmodels.tsa.arima.model import ARIMA

        # 拟合模型
        model = ARIMA(ts, order=order)
        result = model.fit()
        print(result.summary())

        # 预测
        forecast = result.forecast(steps=forecast_steps)

        # 可视化
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(ts.iloc[-180:], label='历史数据')
        forecast_index = pd.date_range(start=ts.index[-1] + timedelta(days=1),
                                       periods=forecast_steps, freq='D')
        ax.plot(forecast_index, forecast, color='red', label=f'ARIMA{order}预测')
        # 置信区间
        ci = result.get_forecast(steps=forecast_steps).conf_int()
        ax.fill_between(forecast_index, ci.iloc[:, 0], ci.iloc[:, 1],
                        color='red', alpha=0.1)
        ax.set_title(f'ARIMA{order}预测')
        ax.legend()
        plt.savefig('ARIMA预测.png', dpi=150, bbox_inches='tight')
        plt.show()

        return result
    except ImportError:
        print("  [提示] 需安装statsmodels: pip install statsmodels")
        return None


# ============================================================
# 4. 指数平滑法
# ============================================================
def demo_exponential_smoothing(ts, forecast_steps=30):
    """Holt-Winters指数平滑"""
    print("\n--- 指数平滑法 ---")
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        # 三种平滑
        models = {
            '简单指数平滑': ExponentialSmoothing(ts, trend=None, seasonal=None),
            'Holt线性趋势': ExponentialSmoothing(ts, trend='add', seasonal=None),
            'Holt-Winters': ExponentialSmoothing(ts, trend='add', seasonal='add', seasonal_periods=365),
        }

        fig, axes = plt.subplots(len(models), 1, figsize=(12, 12))
        for ax, (name, model) in zip(axes, models.items()):
            fit = model.fit()
            forecast = fit.forecast(forecast_steps)
            forecast_index = pd.date_range(start=ts.index[-1] + timedelta(days=1),
                                           periods=forecast_steps, freq='D')
            ax.plot(ts.iloc[-180:], label='历史数据')
            ax.plot(forecast_index, forecast, color='red', label=f'{name}预测')
            ax.set_title(name)
            ax.legend()

        plt.tight_layout()
        plt.savefig('指数平滑法.png', dpi=150, bbox_inches='tight')
        plt.show()
    except ImportError:
        print("  [提示] 需安装statsmodels: pip install statsmodels")


# ============================================================
# 5. 差分与平稳化
# ============================================================
def make_stationary(ts):
    """差分使时间序列平稳"""
    print("\n--- 差分平稳化 ---")
    ts_diff = ts.diff().dropna()
    ts_diff2 = ts.diff().diff().dropna()

    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    axes[0].plot(ts, linewidth=0.8)
    axes[0].set_title('原始序列')
    axes[1].plot(ts_diff, linewidth=0.8, color='orange')
    axes[1].set_title('一阶差分')
    axes[2].plot(ts_diff2, linewidth=0.8, color='green')
    axes[2].set_title('二阶差分')
    plt.tight_layout()
    plt.savefig('差分平稳化.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ADF检验
    try:
        from statsmodels.tsa.stattools import adfuller
        for name, data in [('原始', ts), ('一阶差分', ts_diff), ('二阶差分', ts_diff2)]:
            result = adfuller(data.dropna())
            print(f"  {name}: ADF={result[0]:.4f}, p={result[1]:.4f} "
                  f"{'平稳' if result[1] < 0.05 else '非平稳'}")
    except ImportError:
        pass

    return ts_diff


# ============================================================
# 主程序演示
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("时间序列分析完整流程演示")
    print("=" * 60)

    ts = create_time_series()

    # 1. 基础分析
    basic_analysis(ts)

    # 2. 分解
    decompose_ts(ts)

    # 3. 差分平稳化
    ts_diff = make_stationary(ts)

    # 4. ARIMA
    demo_arima(ts, order=(1, 1, 1))

    # 5. 指数平滑
    demo_exponential_smoothing(ts)
