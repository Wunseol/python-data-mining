# Python 数据挖掘项目规则

## 项目概述

Python 数据挖掘学习路径项目，系统化覆盖数据挖掘知识体系，遵循 CRISP-DM 标准流程。

## 技术栈

- Python 3.8+
- 核心依赖：numpy, pandas, matplotlib, scikit-learn, scipy
- 可选依赖：tensorflow, h5py, jieba, networkx

## 编码规范

- 遵循 PEP 8 代码风格
- 函数/变量用 snake_case，类用 PascalCase
- 关键算法含详细中文注释
- 文件编码 UTF-8
- 导入顺序：标准库 → 第三方库 → 本地模块
- 使用 `# ====...====` 分节标记
- 每个模块可独立运行，无跨模块硬编码依赖
- 随机种子固定 `random_state=42` 或 `np.random.seed(42)`

## 运行与测试命令

### 语法检查
```bash
python -m py_compile "模块路径/文件名.py"
```

### Lint 检查
```bash
flake8 --max-line-length=120 --exclude=.git,__pycache__,venv,docs .
```

### 类型检查
```bash
mypy --ignore-missing-imports --no-strict-optional .
```

### 单元测试
```bash
pytest tests/ -v
```

## 项目结构约定

- 顶级目录：`编号_模块名`（如 `03_回归分析/`）
- 子目录：`编号_子方向名`（如 `01_聚类分析/`）
- 核心算法文件：`中文名称.py`（如 `K近邻算法.py`）
- 案例文件：`编号_案例名称.py`（如 `02_手写数字识别.py`）
- 辅助模块：`英文名称.py`（如 `treePlotter.py`）

## 注意事项

- Matplotlib 中文字体：`plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']`
- 示例数据应内嵌代码，使用 sklearn 内置数据集或合成数据
- `plt.show()` 在无 GUI 环境下需使用 `Agg` 后端
