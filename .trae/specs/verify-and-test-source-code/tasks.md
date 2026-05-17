# Tasks

- [x] Task 1: 修复 Python 2/3 兼容性问题
  - [x] 1.1 regTrees.py: `map(float,curLine)` → `list(map(float,curLine))`
  - [x] 1.2 5处 `== None` / `!= None` → `is None` / `is not None`（regTrees.py, CART.py×2, fpGrowth.py, FP_Growth.py）
  - [x] 1.3 4处 `canvas.show()` → `canvas.draw()`（treeExplore.py×2, test/GUI.py×2）

- [x] Task 2: 修复资源泄漏（`fr=open()` → `with open()`）
  - [x] 2.1 K近邻算法.py: 3处 open()
  - [x] 2.2 SVM算法.py: 2处 open()
  - [x] 2.3 trees.py: 2处 open()
  - [x] 2.4 regTrees.py: 1处 open()
  - [x] 2.5 CART.py (03_CART回归树): 1处 open()
  - [x] 2.6 CART.py (04_CART可视化GUI): 1处 open()
  - [x] 2.7 ID3分类案例.py: 1处 open()
  - [x] 2.8 test/regTrees.py: 1处 open()
  - [x] 2.9 PCA.py: 1处 open()

- [x] Task 3: 消除通配符导入
  - [x] 3.1 treeExplore.py: `from numpy import *` → 显式导入
  - [x] 3.2 GUI.py (04_CART可视化GUI): `from numpy import *` → 显式导入
  - [x] 3.3 test/regTrees.py: `from numpy import *` → 显式导入
  - [x] 3.4 treeExplore.py: `from tkinter import *` → 显式导入
  - [x] 3.5 GUI.py: `from tkinter import *` → 显式导入
  - [x] 3.6 test/GUI.py: `from tkinter import *` → 显式导入

- [x] Task 4: 统一中文字体设置（14个文件 `plt.rcParams` → `setup_chinese_font`）
  - [x] 4.1 数据仓库基础.py
  - [x] 4.2 数据可视化.py
  - [x] 4.3 01_线性回归.py
  - [x] 4.4 02_逻辑回归.py
  - [x] 4.5 半监督学习与迁移学习.py
  - [x] 4.6 01_模型评估与调优.py
  - [x] 4.7 02_类别不平衡处理.py
  - [x] 4.8 KMeans聚类.py
  - [x] 4.9 异常检测.py
  - [x] 4.10 神经网络基础.py
  - [x] 4.11 NLP基础.py
  - [x] 4.12 时间序列分析.py
  - [x] 4.13 图与网络挖掘.py
  - [x] 4.14 流数据挖掘.py

- [x] Task 5: 补充 docstring 和 `__main__`（核心文件）
  - [x] 5.1 K近邻算法.py: 补充模块级 docstring
  - [x] 5.2 朴素贝叶斯算法.py: 补充模块级 docstring
  - [x] 5.3 ID3补充实现.py: 补充 docstring
  - [x] 5.4 C45决策树.py: 补充模块级 docstring
  - [x] 5.5 SVM算法.py: 补充模块级 docstring
  - [x] 5.6 Apriori.py: 补充模块级 docstring
  - [x] 5.7 PCA.py: 补充 docstring + `__main__`
  - [x] 5.8 SVD图像压缩/SVD.py: 补充 docstring
  - [x] 5.9 08_深度学习/02_文本分类模型对比: 9个文件补充 docstring + `__main__`

- [x] Task 6: 编写单元测试
  - [x] 6.1 test_04_分类算法.py: 17个测试
  - [x] 6.2 test_05_模型评估与调优.py: 15个测试
  - [x] 6.3 test_06_集成学习.py: 11个测试
  - [x] 6.4 test_07_无监督学习.py: 11个测试
  - [x] 6.5 test_08_深度学习.py: 14个测试
  - [x] 6.6 test_09_应用领域.py: 12个测试

- [x] Task 7: 全面验证
  - [x] 7.1 77个源文件 py_compile 全部通过
  - [x] 7.2 pytest 119个测试全部通过
  - [x] 7.3 flake8 无严重功能性问题
  - [x] 7.4 6种禁止模式全部验证通过

# Task Dependencies
- Task 2 depends on Task 1 (先修兼容性再修资源泄漏)
- Task 3 is independent
- Task 4 is independent
- Task 5 depends on Task 4 (先统一字体设置再补 docstring)
- Task 6 depends on Tasks 1-5 (代码修复完成后再写测试)
- Task 7 depends on all other tasks
