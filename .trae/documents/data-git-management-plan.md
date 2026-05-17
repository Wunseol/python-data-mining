# 数据文件 Git 管理策略计划

## 目标

确保所有数据文件的**压缩版（.zip）被 Git 跟踪**，**未压缩版被 .gitignore 排除**，同时保证本地代码可以正常运行（未压缩版存在于本地）。

## 当前状态

| 数据 | 压缩版(.zip) | 未压缩版 | 需要操作 |
|------|:---:|:---:|------|
| KNN digits | ✅ digits.zip | ❌ 已删除 | 解压恢复 → ignore 未压缩目录 |
| 朴素贝叶斯 email | ✅ email.zip | ❌ 已删除 | 解压恢复 → ignore 未压缩目录 |
| SVM digits | ✅ digits.zip | ❌ 已删除 | 解压恢复 → ignore 未压缩目录 |
| kosarak.dat | ✅ kosarak.zip | ✅ 在 kosarak/ 子目录 | 移出到正确位置 → ignore .dat |
| secom.data | ✅ secom.zip | ✅ 在 secom/ 子目录 | 移出到正确位置 → ignore .data |
| mushroom.dat | ❌ 未压缩 | ✅ 直接在目录中 | 压缩为 .zip → ignore .dat |
| data.zip (深度学习) | ✅ data.zip | ❌ 不存在 | 解压恢复 data/ 目录 → ignore |
| machinelearninginaction-master/ | — | 整个目录 | ignore 整个目录 |

## 实施步骤

### 步骤1：解压已删除的未压缩数据（3个目录 + 1个深度学习数据）

1.1 解压 KNN digits.zip
- 在 `04_分类算法/01_K近邻算法/` 下解压 `digits.zip`
- 解压后应得到 `digits/trainingDigits/` 和 `digits/testDigits/`

1.2 解压朴素贝叶斯 email.zip
- 在 `04_分类算法/02_朴素贝叶斯/` 下解压 `email.zip`
- 解压后应得到 `email/ham/` 和 `email/spam/`

1.3 解压 SVM digits.zip
- 在 `04_分类算法/04_支持向量机/` 下解压 `digits.zip`
- 解压后应得到 `trainingDigits/` 和 `testDigits/`

1.4 解压深度学习 data.zip
- 在 `08_深度学习/02_文本分类模型对比/` 下解压 `data.zip`
- 解压后应得到 `data/train_onehot.hdf5` 和 `data/test_onehot.hdf5`

### 步骤2：修正 kosarak.dat 和 secom.data 的位置

2.1 修正 kosarak.dat
- 当前位置：`07_无监督学习/02_关联规则挖掘/02_FPGrowth算法/kosarak/kosarak.dat`
- 目标位置：`07_无监督学习/02_关联规则挖掘/02_FPGrowth算法/kosarak.dat`
- 操作：将文件从 kosarak/ 子目录移出到父目录，然后删除空的 kosarak/ 子目录

2.2 修正 secom.data
- 当前位置：`07_无监督学习/03_降维与矩阵分解/01_PCA主成分分析/secom/secom.data`
- 目标位置：`07_无监督学习/03_降维与矩阵分解/01_PCA主成分分析/secom.data`
- 操作：将文件从 secom/ 子目录移出到父目录，然后删除空的 secom/ 子目录

### 步骤3：压缩 mushroom.dat

- 在 `07_无监督学习/02_关联规则挖掘/01_Apriori算法/` 下将 `mushroom.dat` 压缩为 `mushroom.zip`
- 保留原始 mushroom.dat（本地运行需要）

### 步骤4：更新 .gitignore

在 `.gitignore` 中添加以下规则，忽略所有未压缩的数据文件/目录，但保留 .zip 文件：

```gitignore
# ---- 数据文件（压缩版跟踪，未压缩版忽略）----
# KNN 手写数字
04_分类算法/01_K近邻算法/digits/

# 朴素贝叶斯邮件
04_分类算法/02_朴素贝叶斯/email/

# SVM 手写数字
04_分类算法/04_支持向量机/trainingDigits/
04_分类算法/04_支持向量机/testDigits/

# 关联规则挖掘数据
07_无监督学习/02_关联规则挖掘/01_Apriori算法/mushroom.dat
07_无监督学习/02_关联规则挖掘/02_FPGrowth算法/kosarak.dat

# PCA 降维数据
07_无监督学习/03_降维与矩阵分解/01_PCA主成分分析/secom.data

# 深度学习文本分类数据
08_深度学习/02_文本分类模型对比/data/

# Machine Learning in Action 源码
machinelearninginaction-master/
```

### 步骤5：更新 docs/数据文件说明.md

更新文档中的解压说明，添加"首次使用需解压"的提示，确保用户知道：
- 克隆仓库后只有 .zip 文件
- 需要解压才能运行对应模块
- .gitignore 已配置忽略未压缩版，不用担心误提交

### 步骤6：验证

- 确认所有 .zip 文件存在
- 确认所有未压缩数据文件/目录存在（本地可运行）
- 确认 .gitignore 规则正确（git status 不显示未压缩数据文件）
