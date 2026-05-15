"""
数据可视化模块 - Data Visualization
====================================
涵盖数据挖掘中常见的可视化技术：
1. 基础图表 (折线、柱状、饼图)
2. 统计图表 (直方图、箱线图、小提琴图)
3. 关系图表 (散点、热力图、气泡图)
4. 高级图表 (雷达图、平行坐标、成对图)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 辅助函数：生成示例数据
# ============================================================
def create_sample_data():
    """创建多维度示例数据"""
    np.random.seed(42)
    n = 300
    data = {
        '年龄': np.random.randint(18, 70, n),
        '收入': np.random.normal(50000, 20000, n),
        '消费': np.random.normal(30000, 10000, n),
        '评分': np.random.uniform(1, 10, n),
        '城市': np.random.choice(['北京', '上海', '广州', '深圳', '成都'], n),
        '性别': np.random.choice(['男', '女'], n),
    }
    df = pd.DataFrame(data)
    df['收入'] = df['收入'].clip(lower=10000)
    df['消费'] = df['消费'].clip(lower=5000)
    return df


# ============================================================
# 1. 基础图表
# ============================================================
def plot_basic_charts(df):
    """基础图表：折线图、柱状图、饼图"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 折线图 — 收入趋势(按年龄排序)
    df_sorted = df.sort_values('年龄')
    age_groups = df_sorted.groupby('年龄')['收入'].mean()
    axes[0].plot(age_groups.index, age_groups.values, 'b-o', markersize=3, linewidth=1)
    axes[0].fill_between(age_groups.index, age_groups.values, alpha=0.2)
    axes[0].set_xlabel('年龄')
    axes[0].set_ylabel('平均收入')
    axes[0].set_title('各年龄段平均收入')

    # 柱状图 — 各城市平均收入
    city_income = df.groupby('城市')['收入'].mean().sort_values(ascending=True)
    colors = plt.cm.Set2(np.linspace(0, 1, len(city_income)))
    axes[1].barh(city_income.index, city_income.values, color=colors)
    axes[1].set_xlabel('平均收入')
    axes[1].set_title('各城市平均收入')

    # 饼图 — 城市分布
    city_counts = df['城市'].value_counts()
    axes[2].pie(city_counts.values, labels=city_counts.index, autopct='%1.1f%%',
                startangle=90, colors=plt.cm.Set3(np.linspace(0, 1, len(city_counts))))
    axes[2].set_title('城市分布')

    plt.tight_layout()
    plt.savefig('基础图表.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 2. 统计图表
# ============================================================
def plot_statistical_charts(df):
    """统计图表：直方图、箱线图、小提琴图"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 直方图 — 收入分布
    axes[0, 0].hist(df['收入'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(df['收入'].mean(), color='red', linestyle='--', label=f"均值={df['收入'].mean():.0f}")
    axes[0, 0].set_xlabel('收入')
    axes[0, 0].set_ylabel('频数')
    axes[0, 0].set_title('收入分布直方图')
    axes[0, 0].legend()

    # 直方图 — 评分分布
    axes[0, 1].hist(df['评分'], bins=20, color='coral', edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('评分')
    axes[0, 1].set_title('评分分布直方图')

    # 分组直方图 — 男女收入对比
    for gender, color in [('男', 'steelblue'), ('女', 'coral')]:
        axes[0, 2].hist(df[df['性别'] == gender]['收入'], bins=25, alpha=0.5,
                        color=color, label=gender, edgecolor='black')
    axes[0, 2].set_xlabel('收入')
    axes[0, 2].set_title('男女收入分布对比')
    axes[0, 2].legend()

    # 箱线图 — 各城市收入
    city_data = [df[df['城市'] == city]['收入'].values for city in df['城市'].unique()]
    bp = axes[1, 0].boxplot(city_data, labels=df['城市'].unique(), patch_artist=True)
    for patch, color in zip(bp['boxes'], plt.cm.Set2(np.linspace(0, 1, len(city_data)))):
        patch.set_facecolor(color)
    axes[1, 0].set_ylabel('收入')
    axes[1, 0].set_title('各城市收入箱线图')

    # 小提琴图 — 各城市评分
    for i, city in enumerate(df['城市'].unique()):
        parts = axes[1, 1].violinplot([df[df['城市'] == city]['评分'].values],
                                       positions=[i], showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_alpha(0.7)
    axes[1, 1].set_xticks(range(len(df['城市'].unique())))
    axes[1, 1].set_xticklabels(df['城市'].unique())
    axes[1, 1].set_ylabel('评分')
    axes[1, 1].set_title('各城市评分小提琴图')

    # 堆叠面积图
    age_bins = pd.cut(df['年龄'], bins=5)
    pivot = df.pivot_table(index=age_bins, columns='性别', values='收入', aggfunc='mean')
    pivot.plot.area(ax=axes[1, 2], alpha=0.6)
    axes[1, 2].set_xlabel('年龄区间')
    axes[1, 2].set_ylabel('平均收入')
    axes[1, 2].set_title('男女平均收入随年龄变化')

    plt.tight_layout()
    plt.savefig('统计图表.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 3. 关系图表
# ============================================================
def plot_relation_charts(df):
    """关系图表：散点图、热力图、气泡图"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 散点图 — 收入vs消费
    for gender, color, marker in [('男', 'steelblue', 'o'), ('女', 'coral', '^')]:
        mask = df['性别'] == gender
        axes[0].scatter(df.loc[mask, '收入'], df.loc[mask, '消费'],
                        c=color, marker=marker, alpha=0.5, s=30, label=gender)
    # 添加趋势线
    z = np.polyfit(df['收入'], df['消费'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['收入'].min(), df['收入'].max(), 100)
    axes[0].plot(x_line, p(x_line), 'r--', linewidth=2, label='趋势线')
    axes[0].set_xlabel('收入')
    axes[0].set_ylabel('消费')
    axes[0].set_title('收入 vs 消费')
    axes[0].legend()

    # 热力图 — 相关系数矩阵
    numeric_cols = ['年龄', '收入', '消费', '评分']
    corr = df[numeric_cols].corr()
    im = axes[1].imshow(corr, cmap='RdYlBu_r', vmin=-1, vmax=1)
    axes[1].set_xticks(range(len(numeric_cols)))
    axes[1].set_yticks(range(len(numeric_cols)))
    axes[1].set_xticklabels(numeric_cols, rotation=45)
    axes[1].set_yticklabels(numeric_cols)
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            axes[1].text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center',
                         fontsize=10, color='white' if abs(corr.iloc[i, j]) > 0.5 else 'black')
    plt.colorbar(im, ax=axes[1])
    axes[1].set_title('相关系数热力图')

    # 气泡图 — 收入vs消费vs评分
    scatter = axes[2].scatter(df['收入'], df['消费'], s=df['评分'] * 15,
                              c=df['年龄'], cmap='viridis', alpha=0.6, edgecolors='black', linewidth=0.5)
    axes[2].set_xlabel('收入')
    axes[2].set_ylabel('消费')
    axes[2].set_title('收入 vs 消费 (气泡=评分, 颜色=年龄)')
    plt.colorbar(scatter, ax=axes[2], label='年龄')

    plt.tight_layout()
    plt.savefig('关系图表.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 4. 高级图表
# ============================================================
def plot_radar_chart(df):
    """雷达图 — 多维对比"""
    cities = df['城市'].unique()
    metrics = ['年龄', '收入', '消费', '评分']

    # 标准化
    city_means = df.groupby('城市')[metrics].mean()
    city_normalized = (city_means - city_means.min()) / (city_means.max() - city_means.min())

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = plt.cm.Set2(np.linspace(0, 1, len(cities)))

    for (city, values), color in zip(city_normalized.iterrows(), colors):
        values_list = values.tolist() + values.tolist()[:1]
        ax.plot(angles, values_list, 'o-', linewidth=2, color=color, label=city)
        ax.fill(angles, values_list, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_title('各城市多维度雷达图', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.savefig('雷达图.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_pair_plot(df):
    """成对散点图 (手动实现)"""
    numeric_cols = ['年龄', '收入', '消费', '评分']
    n = len(numeric_cols)

    fig, axes = plt.subplots(n, n, figsize=(14, 14))
    colors = {'男': 'steelblue', '女': 'coral'}

    for i in range(n):
        for j in range(n):
            ax = axes[i][j]
            if i == j:
                # 对角线：直方图
                for gender, color in colors.items():
                    ax.hist(df[df['性别'] == gender][numeric_cols[i]],
                            bins=20, alpha=0.5, color=color, edgecolor='black')
            else:
                # 非对角线：散点图
                for gender, color in colors.items():
                    mask = df['性别'] == gender
                    ax.scatter(df.loc[mask, numeric_cols[j]], df.loc[mask, numeric_cols[i]],
                               c=color, s=8, alpha=0.4)

            if i == n - 1:
                ax.set_xlabel(numeric_cols[j])
            if j == 0:
                ax.set_ylabel(numeric_cols[i])

    plt.suptitle('成对散点图矩阵', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('成对散点图.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 主程序演示
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("数据可视化完整流程演示")
    print("=" * 60)

    df = create_sample_data()
    print(f"数据集: {df.shape[0]}行, {df.shape[1]}列")
    print(df.head())

    # 1. 基础图表
    print("\n--- 1. 基础图表 ---")
    plot_basic_charts(df)

    # 2. 统计图表
    print("\n--- 2. 统计图表 ---")
    plot_statistical_charts(df)

    # 3. 关系图表
    print("\n--- 3. 关系图表 ---")
    plot_relation_charts(df)

    # 4. 高级图表
    print("\n--- 4. 雷达图 ---")
    plot_radar_chart(df)

    print("\n--- 5. 成对散点图 ---")
    plot_pair_plot(df)
