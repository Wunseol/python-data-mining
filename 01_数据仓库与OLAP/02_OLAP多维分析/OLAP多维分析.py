"""
OLAP多维分析模块 - OLAP Multidimensional Analysis
==================================================
涵盖OLAP的核心操作与实现：
1. 数据立方体构建与计算
2. OLAP基本操作（上卷、下钻、切片、切块、旋转）
3. OLAP服务器架构（ROLAP/MOLAP/HOLAP）
4. 聚合与预计算策略
5. 数据泛化：面向属性的归纳

参考：Han & Kamber《数据挖掘：概念与技术》第4-5章
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from utils import setup_chinese_font

import pandas as pd
import numpy as np
from itertools import product

import matplotlib.pyplot as plt
setup_chinese_font()


# ============================================================
# 1. 数据立方体构建
# ============================================================
def build_data_cube():
    """构建数据立方体并展示多维聚合"""

    # 原始销售数据
    sales_data = pd.DataFrame({
        '年份': [2023, 2023, 2023, 2023, 2024, 2024, 2024, 2024],
        '季度': ['Q1', 'Q1', 'Q2', 'Q2', 'Q1', 'Q1', 'Q2', 'Q2'],
        '地区': ['北京', '上海', '北京', '上海', '北京', '上海', '北京', '上海'],
        '产品': ['电脑', '电脑', '手机', '手机', '电脑', '电脑', '手机', '手机'],
        '销售额': [120, 80, 200, 150, 140, 90, 220, 180],
        '销量': [12, 8, 40, 30, 14, 9, 44, 36],
    })

    print("=" * 60)
    print("原始销售数据")
    print("=" * 60)
    print(sales_data)

    # 构建多维数据立方体 - 各级聚合
    print("\n" + "=" * 60)
    print("数据立方体 - 多级聚合")
    print("=" * 60)

    # 0维聚合（总合计）
    total = sales_data[['销售额', '销量']].sum()
    print(f"\n0维(总合计): 销售额={total['销售额']}, 销量={total['销量']}")

    # 1维聚合
    for dim in ['年份', '季度', '地区', '产品']:
        agg = sales_data.groupby(dim)[['销售额', '销量']].sum()
        print(f"\n1维(按{dim}):")
        print(agg)

    # 2维聚合
    for d1, d2 in [('年份', '地区'), ('年份', '产品'), ('地区', '产品')]:
        agg = sales_data.groupby([d1, d2])[['销售额', '销量']].sum()
        print(f"\n2维(按{d1}×{d2}):")
        print(agg)

    return sales_data


# ============================================================
# 2. OLAP基本操作
# ============================================================
def demonstrate_olap_operations(sales_data):
    """演示OLAP五大基本操作"""

    print("\n" + "=" * 60)
    print("OLAP基本操作")
    print("=" * 60)

    # --- 上卷 (Roll-up) ---
    print("\n--- 上卷 (Roll-up): 沿维度层次向上聚合 ---")
    print("操作: 地区→全部 (消除地区维度)")
    rollup = sales_data.groupby(['年份', '季度', '产品'])[['销售额', '销量']].sum()
    print(rollup)

    print("\n操作: 季度→年份 (沿时间层次上卷)")
    rollup2 = sales_data.groupby(['年份', '地区', '产品'])[['销售额', '销量']].sum()
    print(rollup2)

    # --- 下钻 (Drill-down) ---
    print("\n--- 下钻 (Drill-down): 沿维度层次向下展开 ---")
    print("操作: 年份→季度 (沿时间层次下钻)")
    drilldown = sales_data.groupby(['年份', '季度', '地区', '产品'])[['销售额', '销量']].sum()
    print(drilldown)

    # --- 切片 (Slice) ---
    print("\n--- 切片 (Slice): 在某一维度上选择一个值 ---")
    print("操作: 选择年份=2024")
    slice_result = sales_data[sales_data['年份'] == 2024]
    print(slice_result)

    # --- 切块 (Dice) ---
    print("\n--- 切块 (Dice): 在多个维度上选择子集 ---")
    print("操作: 年份=2024 AND 地区=北京 AND 产品=电脑")
    dice_result = sales_data[
        (sales_data['年份'] == 2024) &
        (sales_data['地区'] == '北京') &
        (sales_data['产品'] == '电脑')
    ]
    print(dice_result)

    # --- 旋转 (Pivot) ---
    print("\n--- 旋转 (Pivot): 重新排列维度方向 ---")
    pivot_result = sales_data.pivot_table(
        values='销售额', index='年份', columns='地区', aggfunc='sum'
    )
    print(pivot_result)


# ============================================================
# 3. OLAP服务器架构
# ============================================================
def demonstrate_olap_architectures():
    """展示OLAP服务器的三种架构"""

    print("\n" + "=" * 60)
    print("OLAP服务器架构对比")
    print("=" * 60)

    comparison = pd.DataFrame({
        '架构': ['ROLAP', 'MOLAP', 'HOLAP'],
        '全称': ['Relational OLAP', 'Multidimensional OLAP', 'Hybrid OLAP'],
        '存储方式': ['关系数据库', '多维数组(MOLAP立方体)', '混合存储'],
        '数据规模': ['适合大数据量', '适合中等数据量', '兼顾大小数据量'],
        '查询速度': ['较慢(需SQL聚合)', '极快(预计算)', '快(热点预计算)'],
        '存储效率': ['高(不预计算)', '低(需预计算所有组合)', '中等'],
        '灵活性': ['高(无维度限制)', '低(维度爆炸问题)', '中'],
        '典型产品': ['MicroStrategy', 'Essbase', 'SQL Server SSAS'],
    })

    print(comparison.to_string(index=False))


# ============================================================
# 4. 聚合与预计算策略
# ============================================================
def demonstrate_precomputation_strategies():
    """演示数据立方体的预计算策略"""

    print("\n" + "=" * 60)
    print("数据立方体预计算策略")
    print("=" * 60)

    # 计算立方体空间
    dimensions = {
        '时间': 4,     # 年×季
        '地区': 2,     # 北京, 上海
        '产品': 2,     # 电脑, 手机
    }

    # 完全立方体的方体数量
    n = len(dimensions)
    num_cuboids = 2 ** n  # 每个维度取或不取
    print(f"\n维度数: {n}")
    print(f"完全立方体的方体数: 2^{n} = {num_cuboids}")
    print(f"  基本方体(0维): 1")
    print(f"  1维方体: {n}")
    print(f"  2维方体: {n * (n-1) // 2}")
    print(f"  顶点方体({n}维): 1")

    # 预计算策略对比
    print("\n预计算策略:")
    strategies = {
        "完全物化": "预计算所有方体，查询最快，但空间开销大",
        "冰山立方体": "只物化满足最小支持度阈值的方体，节省空间",
        "闭立方体": "只物化闭项集对应的方体，压缩冗余",
        "立方体外壳": "只预计算低维(如2-3维)方体，高维按需计算",
        "壳片段": "预计算部分维度的片段，查询时动态组合",
    }
    for name, desc in strategies.items():
        print(f"  · {name}: {desc}")


# ============================================================
# 5. 数据泛化：面向属性的归纳
# ============================================================
def demonstrate_attribute_oriented_induction():
    """演示面向属性的归纳（AOI）"""

    print("\n" + "=" * 60)
    print("面向属性的归纳 (Attribute-Oriented Induction)")
    print("=" * 60)

    # 原始数据
    data = pd.DataFrame({
        '姓名': ['张三', '李四', '王五', '赵六', '钱七', '孙八', '周九', '吴十'],
        '性别': ['男', '女', '男', '女', '男', '男', '女', '男'],
        '年龄': [25, 35, 45, 28, 52, 31, 40, 60],
        '职位': ['程序员', '经理', '总监', '程序员', '总监', '经理', '程序员', '顾问'],
        '薪资': [15000, 25000, 40000, 18000, 45000, 28000, 16000, 50000],
        '部门': ['研发', '市场', '研发', '研发', '管理', '市场', '研发', '管理'],
    })

    print("\n原始数据:")
    print(data)

    # 概念层次泛化
    print("\n概念层次泛化:")
    print("  年龄: 具体值 → 青年(≤35) / 中年(36-50) / 老年(>50)")
    print("  薪资: 具体值 → 低(≤20K) / 中(20K-35K) / 高(>35K)")
    print("  职位: 具体值 → 执行层(程序员) / 管理层(经理/总监) / 咨询层(顾问)")

    # 执行泛化
    generalized = data.copy()
    generalized['年龄段'] = pd.cut(
        generalized['年龄'], bins=[0, 35, 50, 100],
        labels=['青年', '中年', '老年']
    )
    generalized['薪资段'] = pd.cut(
        generalized['薪资'], bins=[0, 20000, 35000, 100000],
        labels=['低薪', '中薪', '高薪']
    )
    generalized['职位层'] = generalized['职位'].map({
        '程序员': '执行层', '经理': '管理层',
        '总监': '管理层', '顾问': '咨询层'
    })

    # 去除原始低层属性，合并相同行
    result = generalized.groupby(
        ['性别', '年龄段', '职位层', '部门']
    ).agg(count=('姓名', 'count')).reset_index()

    print("\n泛化后的结果（合并等价行）:")
    print(result)

    # 特征化规则提取
    print("\n特征化规则:")
    total = len(data)
    for _, row in result.iterrows():
        if row['count'] >= 2:  # 只显示频次≥2的规则
            support = row['count'] / total
            print(f"  若 性别={row['性别']} ∧ 职位层={row['职位层']} "
                  f"∧ 部门={row['部门']} → count={row['count']} "
                  f"(支持度={support:.1%})")


def visualize_olap_operations(sales_data):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax1 = axes[0, 0]
    slice_2023 = sales_data[sales_data['年份'] == 2023].groupby('地区')['销售额'].sum()
    slice_2024 = sales_data[sales_data['年份'] == 2024].groupby('地区')['销售额'].sum()
    x = np.arange(len(slice_2023.index))
    width = 0.35
    ax1.bar(x - width / 2, slice_2023.values, width, label='2023', color='#3498db', alpha=0.85)
    ax1.bar(x + width / 2, slice_2024.values, width, label='2024', color='#e74c3c', alpha=0.85)
    ax1.set_xlabel('地区')
    ax1.set_ylabel('销售额')
    ax1.set_title('切片：按年份选择（2023 vs 2024）')
    ax1.set_xticks(x)
    ax1.set_xticklabels(slice_2023.index)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    ax2 = axes[0, 1]
    dice_result = sales_data[
        (sales_data['年份'] == 2024) & (sales_data['地区'] == '北京')
    ]
    full_2024_beijing = sales_data[
        (sales_data['年份'] == 2024) & (sales_data['地区'] == '北京')
    ]
    categories = full_2024_beijing['产品'].tolist()
    values = full_2024_beijing['销售额'].tolist()
    colors_dice = ['#2ecc71', '#f39c12']
    bars = ax2.bar(categories, values, color=colors_dice, alpha=0.85, edgecolor='k')
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, str(val), ha='center', fontsize=11)
    ax2.set_xlabel('产品')
    ax2.set_ylabel('销售额')
    ax2.set_title('切块：2024年北京（多维选择子集）')
    ax2.grid(True, alpha=0.3, axis='y')

    ax3 = axes[1, 0]
    by_quarter = sales_data.groupby('季度')['销售额'].sum()
    by_year = sales_data.groupby('年份')['销售额'].sum()
    ax3.bar(['Q1', 'Q2'], by_quarter.values, color='#9b59b6', alpha=0.85, label='按季度（下钻）')
    ax3_twin = ax3.twinx()
    ax3_twin.bar(['2023', '2024'], by_year.values, width=0.4, color='#1abc9c', alpha=0.6, label='按年份（上卷）')
    ax3.set_xlabel('维度层次')
    ax3.set_ylabel('季度销售额')
    ax3_twin.set_ylabel('年度销售额')
    ax3.set_title('上卷/下钻：时间维度层次聚合对比')
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    ax3.grid(True, alpha=0.3, axis='y')

    ax4 = axes[1, 1]
    pivot_result = sales_data.pivot_table(values='销售额', index='年份', columns='地区', aggfunc='sum')
    pivot_result.plot(kind='bar', ax=ax4, color=['#3498db', '#e74c3c'], alpha=0.85, edgecolor='k')
    ax4.set_xlabel('年份')
    ax4.set_ylabel('销售额')
    ax4.set_title('旋转：维度方向重排（年份×地区）')
    ax4.legend(title='地区')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=0)

    plt.tight_layout()
    plt.show()


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════╗")
    print("║     OLAP多维分析 - Multidimensional OLAP     ║")
    print("╚══════════════════════════════════════════════╝")

    sales_data = build_data_cube()
    demonstrate_olap_operations(sales_data)
    demonstrate_olap_architectures()
    demonstrate_precomputation_strategies()
    demonstrate_attribute_oriented_induction()
    visualize_olap_operations(sales_data)
