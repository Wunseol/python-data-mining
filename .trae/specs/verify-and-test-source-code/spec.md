# 项目源码正确性与完备性检查 Spec

## Why
项目经过多轮迭代（内容补全、新技术添加），67个源文件中仍存在 Python 2/3 兼容性问题、资源泄漏、代码风格不一致、缺少 docstring/__main__、测试覆盖不足等问题，需要系统性检查和修复以确保代码的正确性和完备性。

## What Changes
- 修复 Python 2/3 兼容性问题（`map()` 无 `list()`、`== None` → `is None`、`canvas.show()` → `canvas.draw()`）
- 修复资源泄漏（`fr=open()` → `with open()`）
- 消除剩余 `from numpy import *` 和 `from tkinter import *` 通配符导入
- 统一中文字体设置（`plt.rcParams` → `from utils import setup_chinese_font`）
- 补充缺失的 docstring 和 `__main__` 入口
- 为所有核心模块编写单元测试，提升测试覆盖率
- 对所有源文件运行 py_compile 确认语法正确

## Impact
- Affected modules: 00-09 全部10个模块
- Affected code: ~30个 .py 文件需要修改
- Affected tests: tests/ 目录需新增测试文件

## ADDED Requirements

### Requirement: Python 2/3 兼容性修复
系统 SHALL 修复所有 Python 2/3 兼容性问题：
1. `map()` 返回迭代器需包裹 `list()`
2. `== None` / `!= None` 替换为 `is None` / `is not None`
3. `canvas.show()` 替换为 `canvas.draw()`

#### Scenario: 运行含 map() 的代码
- **WHEN** 用户运行 regTrees.py
- **THEN** `map(float, curLine)` 应返回 list 而非 map 对象

### Requirement: 资源泄漏修复
系统 SHALL 将所有裸 `open()` 调用替换为 `with open()` 上下文管理器。

#### Scenario: 运行含文件读取的代码
- **WHEN** 用户运行 K近邻算法.py 或 SVM算法.py
- **THEN** 文件句柄应自动关闭，无资源泄漏

### Requirement: 通配符导入消除
系统 SHALL 消除所有 `from numpy import *` 和 `from tkinter import *` 用法，改为显式导入。

#### Scenario: 运行 flake8 检查
- **WHEN** 对项目运行 flake8
- **THEN** 不应报告 wildcard import 警告

### Requirement: 中文字体设置统一
系统 SHALL 将所有使用手动 `plt.rcParams['font.sans-serif']` 的文件改为使用 `from utils import setup_chinese_font`。

#### Scenario: 运行含可视化的模块
- **WHEN** 用户运行任意含 matplotlib 的模块
- **THEN** 中文字体应正确显示，且使用统一的 setup_chinese_font() 方式

### Requirement: docstring 和 __main__ 补充
系统 SHALL 为所有核心源文件补充模块级 docstring 和 `if __name__ == '__main__':` 入口（辅助模块如 treePlotter.py、GUI 模块除外）。

#### Scenario: 直接运行核心算法文件
- **WHEN** 用户直接运行 PCA.py 或 SVD.py
- **THEN** 应执行主程序演示，而非无输出

### Requirement: 单元测试覆盖
系统 SHALL 为以下核心模块新增单元测试：
1. 04_分类算法（KNN、朴素贝叶斯、决策树）
2. 05_模型评估与调优（可解释AI）
3. 06_集成学习（现代梯度提升）
4. 07_无监督学习（关联规则、异常检测）
5. 08_深度学习（自编码器/VAE、对比学习、Transformer）
6. 09_应用领域（图神经网络、联邦学习）

#### Scenario: 运行 pytest
- **WHEN** 用户运行 `pytest tests/ -v`
- **THEN** 所有测试应通过，测试数量 ≥ 60

## MODIFIED Requirements

### Requirement: NFR1 独立可运行
扩展覆盖范围：所有核心源文件（含 PCA.py、SVD.py 等旧文件）均需可独立运行。

## REMOVED Requirements

无移除需求。
