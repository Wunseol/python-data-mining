"""
数据仓库基础模块 - Data Warehouse Fundamentals
================================================
涵盖数据仓库的核心概念与实现：
1. 数据仓库体系结构（多层架构）
2. 多维数据模型（星形/雪花/事实星座模式）
3. ETL过程（提取-转换-装载）
4. 元数据管理
5. 数据仓库 vs 操作数据库

参考：Han & Kamber《数据挖掘：概念与技术》第4章
"""

import pandas as pd
import numpy as np
from collections import defaultdict

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 数据仓库体系结构
# ============================================================
def demonstrate_warehouse_architecture():
    """展示数据仓库的多层体系结构"""
    architecture = {
        "数据源层": {
            "关系数据库": "企业ERP、CRM等业务系统",
            "_flat文件": "CSV、Excel、日志文件",
            "API接口": "第三方数据服务",
            "实时流": "传感器、IoT设备数据",
        },
        "数据暂存区": {
            "ETL处理": "提取(Extract)-转换(Transform)-装载(Load)",
            "数据清洗": "去重、纠错、一致性检查",
            "数据集成": "多源数据的实体识别与冲突解决",
        },
        "数据仓库层": {
            "企业仓库": "全企业范围的集中式仓库",
            "数据集市": "面向特定部门/主题的子集",
            "虚拟仓库": "操作数据库上的视图集合",
        },
        "分析服务层": {
            "OLAP服务器": "多维数据分析服务(ROLAP/MOLAP/HOLAP)",
            "数据挖掘": "模式发现与知识提取",
            "报表服务": "定期报表与即席查询",
        },
    }

    print("=" * 60)
    print("数据仓库多层体系结构")
    print("=" * 60)
    for layer, components in architecture.items():
        print(f"\n【{layer}】")
        for comp, desc in components.items():
            print(f"  {comp}: {desc}")

    return architecture


# ============================================================
# 2. 多维数据模型
# ============================================================
def demonstrate_multidimensional_model():
    """演示多维数据模型：星形、雪花、事实星座模式"""

    # --- 星形模式 (Star Schema) ---
    print("\n" + "=" * 60)
    print("星形模式 (Star Schema)")
    print("=" * 60)

    # 事实表
    sales_fact = pd.DataFrame({
        'time_key': [1, 1, 2, 2, 3, 3],
        'item_key': [101, 102, 101, 103, 102, 103],
        'branch_key': [1, 1, 2, 2, 1, 2],
        'dollars_sold': [1000, 500, 800, 300, 600, 450],
        'units_sold': [10, 5, 8, 3, 6, 4],
    })
    print("\n事实表(sales_fact):")
    print(sales_fact)

    # 维度表
    time_dim = pd.DataFrame({
        'time_key': [1, 2, 3],
        'day': [1, 2, 3],
        'month': ['Jan', 'Jan', 'Jan'],
        'quarter': ['Q1', 'Q1', 'Q1'],
        'year': [2024, 2024, 2024],
    })
    item_dim = pd.DataFrame({
        'item_key': [101, 102, 103],
        'item_name': ['电脑', '手机', '平板'],
        'brand': ['品牌A', '品牌B', '品牌A'],
        'category': ['电子', '电子', '电子'],
    })
    branch_dim = pd.DataFrame({
        'branch_key': [1, 2],
        'branch_name': ['北京店', '上海店'],
        'region': ['华北', '华东'],
    })

    print("\n维度表(time_dim):")
    print(time_dim)
    print("\n维度表(item_dim):")
    print(item_dim)
    print("\n维度表(branch_dim):")
    print(branch_dim)

    # --- 雪花模式 (Snowflake Schema) ---
    print("\n" + "=" * 60)
    print("雪花模式 (Snowflake Schema) - 维度表进一步规范化")
    print("=" * 60)

    brand_dim = pd.DataFrame({
        'brand_key': [1, 2],
        'brand_name': ['品牌A', '品牌B'],
        'brand_category': ['高端', '中端'],
    })
    item_dim_snow = pd.DataFrame({
        'item_key': [101, 102, 103],
        'item_name': ['电脑', '手机', '平板'],
        'brand_key': [1, 2, 1],  # 外键指向brand_dim
        'category': ['电子', '电子', '电子'],
    })
    print("\n规范化维度表(item_dim_snow) - brand拆分到独立表:")
    print(item_dim_snow)
    print("\n品牌维度表(brand_dim):")
    print(brand_dim)

    # --- 事实星座模式 (Fact Constellation) ---
    print("\n" + "=" * 60)
    print("事实星座模式 (Fact Constellation) - 多个事实表共享维度")
    print("=" * 60)
    print("sales_fact  ←→ 共享 time_dim, item_dim, branch_dim")
    print("shipping_fact ←→ 共享 time_dim, item_dim, + shipper_dim")
    print("适用于企业级数据仓库，多个业务过程各有事实表")


# ============================================================
# 3. ETL过程
# ============================================================
def demonstrate_etl_process():
    """演示ETL（提取-转换-装载）过程"""

    print("\n" + "=" * 60)
    print("ETL过程演示")
    print("=" * 60)

    # --- Extract（提取）---
    print("\n--- Extract: 从多个源系统提取数据 ---")
    source_crm = pd.DataFrame({
        'customer_id': ['C001', 'C002', 'C003'],
        'name': ['张三', '李四', '王五'],
        'phone': ['13800001111', '13800002222', '13800003333'],
    })
    source_erp = pd.DataFrame({
        'cust_id': ['C001', 'C002', 'C004'],
        'customer_name': ['张三', '李四', '赵六'],
        'total_purchase': [5000, 3000, 8000],
    })
    print("CRM系统数据:")
    print(source_crm)
    print("\nERP系统数据:")
    print(source_erp)

    # --- Transform（转换）---
    print("\n--- Transform: 数据清洗、统一格式、冲突解决 ---")

    # 实体识别：customer_id vs cust_id
    # 冲突解决：name vs customer_name 不一致时取最新源
    merged = pd.merge(
        source_crm, source_erp,
        left_on='customer_id', right_on='cust_id',
        how='outer'
    )
    merged['final_name'] = merged['name'].fillna(merged['customer_name'])
    merged['final_id'] = merged['customer_id'].fillna(merged['cust_id'])

    # 数据类型统一与默认值填充
    merged['total_purchase'] = merged['total_purchase'].fillna(0)
    transformed = merged[['final_id', 'final_name', 'phone', 'total_purchase']].copy()
    transformed.columns = ['customer_id', 'name', 'phone', 'total_purchase']

    print("\n转换后的统一数据:")
    print(transformed)

    # --- Load（装载）---
    print("\n--- Load: 装入数据仓库 ---")
    print("装载策略:")
    print("  · 初始装载: 首次全量导入历史数据")
    print("  · 增量装载: 定期导入新增/变更数据")
    print("  · 缓慢变化维度(SCD)处理:")
    print("    - Type 1: 直接覆盖旧值")
    print("    - Type 2: 新增一行记录历史（添加起止时间）")
    print("    - Type 3: 新增一列保存旧值")

    return transformed


# ============================================================
# 4. 元数据管理
# ============================================================
def demonstrate_metadata():
    """演示元数据管理"""
    print("\n" + "=" * 60)
    print("元数据管理")
    print("=" * 60)

    metadata = {
        "技术元数据": {
            "表结构": "列名、数据类型、约束、索引",
            "ETL规则": "映射关系、转换逻辑、调度频率",
            "数据血缘": "字段从源到目标的追踪",
            "存储信息": "分区策略、压缩方式、大小",
        },
        "业务元数据": {
            "业务术语": "指标定义、计算口径",
            "维度层次": "时间→月→季→年的层级关系",
            "数据所有者": "各业务域的责任人",
            "数据质量规则": "完整性、一致性、时效性阈值",
        },
    }

    for category, items in metadata.items():
        print(f"\n【{category}】")
        for key, value in items.items():
            print(f"  {key}: {value}")


# ============================================================
# 5. 数据仓库 vs 操作数据库
# ============================================================
def compare_warehouse_vs_oltp():
    """对比数据仓库与操作数据库(OLTP)"""
    print("\n" + "=" * 60)
    print("数据仓库 vs 操作数据库(OLTP)")
    print("=" * 60)

    comparison = pd.DataFrame({
        '对比维度': ['面向', '数据内容', '数据量', '时间范围', '操作类型',
                     '查询复杂度', '冗余', '用户数', '设计模式'],
        'OLTP(操作数据库)': ['面向事务处理', '当前/最新数据', 'MB~GB', '当前数据',
                           '增删改频繁', '简单快速查询', '最小冗余', '大量并发用户',
                           '实体-关系(ER)'],
        '数据仓库': ['面向分析决策', '历史/汇总数据', 'GB~TB~PB', '历史数据(年)',
                    '以读为主(批量装载)', '复杂聚合查询', '适度冗余(星形模式)',
                    '少数分析人员', '多维数据模型'],
    })

    print(comparison.to_string(index=False))


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════╗")
    print("║        数据仓库基础 - Data Warehouse         ║")
    print("╚══════════════════════════════════════════════╝")

    demonstrate_warehouse_architecture()
    demonstrate_multidimensional_model()
    demonstrate_etl_process()
    demonstrate_metadata()
    compare_warehouse_vs_oltp()
