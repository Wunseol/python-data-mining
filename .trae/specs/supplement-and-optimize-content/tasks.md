# Tasks

- [x] Task 1: 修复 Python 2 语法和缺失 import
  - [x] 1.1 trees.py: `dict.iteritems()` → `dict.items()`
  - [x] 1.2 朴素贝叶斯算法.py: `dict.iteritems()` → `dict.items()`
  - [x] 1.3 01_模型评估与调优.py: 添加 `import pandas as pd`
  - [x] 1.4 特征工程.py: 添加 `import pandas as pd`（文件位于 02_数据探索与处理/01_数据预处理与特征工程/）

- [x] Task 2: 消除 `from numpy import *` 用法（5个文件）
  - [x] 2.1 K近邻算法.py: 替换为显式 numpy 导入
  - [x] 2.2 朴素贝叶斯算法.py: 替换为显式 numpy 导入
  - [x] 2.3 regTrees.py: 替换为显式 numpy 导入
  - [x] 2.4 SVM算法.py: 替换为显式 numpy 导入
  - [x] 2.5 fpGrowth.py: 无需修改（该文件不使用 numpy）

- [x] Task 3: 补充 docstring 和 `__main__` 入口（6个文件）
  - [x] 3.1 K近邻算法.py: 添加 docstring + `__main__` + sklearn 对比
  - [x] 3.2 朴素贝叶斯算法.py: 添加 docstring + `__main__`
  - [x] 3.3 trees.py: 添加 docstring + `__main__`
  - [x] 3.4 C45决策树.py: 添加 docstring
  - [x] 3.5 SVM算法.py: 添加 docstring + `__main__` + sklearn 对比
  - [x] 3.6 fpGrowth.py: 添加 docstring + `__main__`

- [x] Task 4: 移除硬编码路径
  - [x] 4.1 ID3补充实现.py: 移除 `C:\Windows\Fonts\simsun.ttc`，改用 utils.setup_chinese_font()

- [x] Task 5: 清理重复文件
  - [x] 5.1 合并 Apriori.py + Apriori补充实现.py → 统一 Apriori.py（Apriori补充实现.py 已删除）
  - [x] 5.2 合并 fpGrowth.py + FP_Growth.py → 统一 FP_Growth算法.py（原文件用户选择保留）

- [x] Task 6: 添加核心算法手动实现
  - [x] 6.1 集成学习.py: 添加 Bagging 手动实现
  - [x] 6.2 集成学习.py: 添加 AdaBoost 手动实现
  - [x] 6.3 高级聚类.py: 添加 DBSCAN 手动实现

- [x] Task 7: 补充可视化（8个模块）
  - [x] 7.1 00_数据挖掘导论: 距离度量对比图
  - [x] 7.2 01_数据仓库与OLAP: OLAP 操作效果对比
  - [x] 7.3 K近邻算法.py: 决策边界可视化
  - [x] 7.4 朴素贝叶斯算法.py: 分类效果可视化
  - [x] 7.5 SVM算法.py: 决策边界可视化
  - [x] 7.6 关联规则挖掘: 频繁项集/规则可视化
  - [x] 7.7 推荐系统.py: 推荐效果对比图
  - [x] 7.8 Web挖掘.py: PageRank/HITS 结果可视化

- [x] Task 8: 验证
  - [x] 8.1 运行 pytest 确认测试通过（37 passed）
  - [x] 8.2 对修改的文件运行 `python -m py_compile` 确认语法正确
  - [x] 8.3 修复 SVM算法.py 双重 `__main__` 问题（合并为单一入口）

# Task Dependencies
- Task 2 depends on Task 1 (先修语法再改导入)
- Task 3 depends on Task 2 (改完导入再补 docstring)
- Task 5 depends on Task 2, 3 (先修好代码再合并)
- Task 6 is independent
- Task 7 depends on Task 4 (先修好字体路径)
- Task 8 depends on all other tasks
