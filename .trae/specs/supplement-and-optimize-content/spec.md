# 数据挖掘方向内容补全与优化 Spec

## Why
项目各数据挖掘方向内容覆盖不均，部分模块存在 Python 2 遗留代码、缺少可视化、缺少手动实现、缺少 docstring/`__main__` 等问题，需要系统性补全和优化以符合项目规范（NFR1-NFR5）。

## What Changes
- 修复 Python 2 语法（`dict.iteritems()` → `dict.items()`）
- 补充缺失的 `import` 语句
- 为缺少可视化的模块添加可视化输出
- 为缺少手动实现的核心算法添加手动实现
- 为缺少 docstring/`__main__` 的文件补充
- 清理重复文件，合并功能重叠的模块
- 消除 `from numpy import *` 用法
- 移除硬编码路径
- 补充缺失的知识点实现

## Impact
- Affected modules: 00-09 全部10个模块
- Affected code: ~30个 .py 文件需要修改
- Affected docs: docs/数据文件说明.md 可能需要更新

## ADDED Requirements

### Requirement: Python 2 语法修复
系统 SHALL 将所有 `dict.iteritems()` 替换为 `dict.items()`，确保 Python 3 兼容性。

#### Scenario: 运行含 iteritems 的代码
- **WHEN** 用户运行 trees.py 或 朴素贝叶斯算法.py
- **THEN** 不应抛出 AttributeError

### Requirement: 缺失 import 补充
系统 SHALL 为所有使用但未导入的模块添加 import 语句。

#### Scenario: 运行模型评估模块
- **WHEN** 用户运行 01_模型评估与调优.py 的 compare_models 函数
- **THEN** 不应抛出 NameError: name 'pd' is not defined

### Requirement: 核心算法手动实现
系统 SHALL 为以下缺少手动实现的核心算法添加手动实现：
1. 集成学习：Bagging（装袋法）手动实现
2. 集成学习：AdaBoost 手动实现
3. 高级聚类：DBSCAN 手动实现

#### Scenario: 运行集成学习模块
- **WHEN** 用户运行集成学习.py
- **THEN** 能看到 Bagging 和 AdaBoost 的手动实现过程和结果

### Requirement: 可视化补充
系统 SHALL 为以下缺少可视化的模块添加 matplotlib 可视化：
1. 00_数据挖掘导论：距离度量对比图
2. 01_数据仓库与OLAP：数据立方体/OLAP操作效果
3. 04_分类算法/KNN：决策边界可视化
4. 04_分类算法/朴素贝叶斯：分类效果可视化
5. 04_分类算法/SVM：决策边界可视化
6. 07_关联规则挖掘：频繁项集/关联规则可视化
7. 09_推荐系统：推荐效果对比图
8. 09_Web挖掘：PageRank/HITS 结果可视化

#### Scenario: 运行数据挖掘导论
- **WHEN** 用户运行 数据挖掘导论.py
- **THEN** 能看到距离度量对比的可视化图表

### Requirement: 代码风格统一
系统 SHALL 对以下文件消除 `from numpy import *` 用法，改为显式导入：
- K近邻算法.py, 朴素贝叶斯算法.py, regTrees.py, SVM算法.py, fpGrowth.py

#### Scenario: 运行 lint 检查
- **WHEN** 对项目运行 flake8
- **THEN** 不应报告 wildcard import 警告

### Requirement: docstring 和 __main__ 补充
系统 SHALL 为以下缺少 docstring 或 `__main__` 的核心文件补充：
- K近邻算法.py, 朴素贝叶斯算法.py, trees.py, C45决策树.py, SVM算法.py, fpGrowth.py

#### Scenario: 运行核心算法文件
- **WHEN** 用户直接运行 K近邻算法.py
- **THEN** 应执行主程序演示，而非无输出

### Requirement: 重复文件清理
系统 SHALL 合并以下重复文件：
1. Apriori.py + Apriori补充实现.py → 统一为 Apriori.py
2. fpGrowth.py + FP_Growth.py → 统一为 FP_Growth算法.py

#### Scenario: 关联规则模块结构
- **WHEN** 用户查看关联规则目录
- **THEN** 每种算法只有一个主文件，无重复实现

### Requirement: 硬编码路径移除
系统 SHALL 移除 ID3补充实现.py 中的硬编码字体路径 `C:\Windows\Fonts\simsun.ttc`，改用 utils.setup_chinese_font()。

## MODIFIED Requirements

### Requirement: NFR4 手动实现优先
原要求"核心算法先手动实现，再对比 sklearn"，现扩展覆盖范围至集成学习(Bagging/AdaBoost)和高级聚类(DBSCAN)。

### Requirement: NFR5 可视化充分
原要求"每个模块包含充分的 matplotlib 可视化"，现明确列出8个需补充可视化的模块。

## REMOVED Requirements

无移除需求。
