# 数据文件整理与文档更新 Spec

## Why
项目中散布大量数据文件（~2,900个），部分文件体积较大（kosarak.dat 30MB、data.zip 9.6MB、secom.data 5.1MB），且部分数据文件缺失（bikeSpeedVsIq_train/test.txt、testSetRBF.txt 等）。需要全面梳理数据文件，将可压缩的目录打包为 zip，并编写完整的数据文件说明文档，方便用户按需获取和管理数据。

## What Changes
- 全面搜索并记录项目中所有数据文件的位置、大小、用途和引用关系
- 将手写数字数据目录（digits/、testDigits/、trainingDigits/）和邮件数据目录（email/）压缩为 zip 文件
- 更新 `docs/数据文件说明.md`，包含完整的数据文件清单、压缩包说明、使用指南和缺失文件说明
- 标注哪些数据文件已内嵌到代码中（如 spamTestSklearn、testDigitsSklearn），哪些仍需外部文件

## Impact
- Affected docs: `docs/数据文件说明.md`
- Affected data directories:
  - `04_分类算法/01_K近邻算法/digits/` → 压缩为 zip
  - `04_分类算法/02_朴素贝叶斯/email/` → 压缩为 zip
  - `04_分类算法/04_支持向量机/testDigits/` 和 `trainingDigits/` → 压缩为 zip
  - `07_无监督学习/02_关联规则挖掘/02_FPGrowth算法/kosarak.dat` → 已是单文件，需用户手动处理
  - `07_无监督学习/02_关联规则挖掘/01_Apriori算法/mushroom.dat` → 已是单文件
  - `07_无监督学习/03_降维与矩阵分解/01_PCA主成分分析/secom.data` → 已是单文件
  - `08_深度学习/02_文本分类模型对比/data.zip` → 已是 zip
  - `07_无监督学习/03_降维与矩阵分解/03_SVD图像压缩/0_5.txt` → 小文件，保留原样
  - `04_分类算法/03_决策树/01_ID3决策树/lenses.txt` → 小文件，保留原样
  - `04_分类算法/03_决策树/04_CART可视化GUI/sine.txt` → 小文件，保留原样

## ADDED Requirements

### Requirement: 数据文件压缩打包
系统 SHALL 将以下数据目录压缩为 zip 文件（放在原目录下），并删除原始文件目录：
1. `04_分类算法/01_K近邻算法/digits/` → `digits.zip`
2. `04_分类算法/02_朴素贝叶斯/email/` → `email.zip`
3. `04_分类算法/04_支持向量机/trainingDigits/` + `testDigits/` → `digits.zip`

#### Scenario: 压缩成功
- **WHEN** 执行压缩操作
- **THEN** 在原目录下生成 zip 文件，原始目录被删除
- **AND** zip 文件大小应小于原始目录总大小

#### Scenario: 用户手动压缩（大文件/单文件）
- **WHEN** 文件为 kosarak.dat (30MB)、secom.data (5MB)、mushroom.dat (557KB) 等单文件
- **THEN** 提示用户手动处理（这些文件不需要压缩）

### Requirement: 数据文件文档更新
系统 SHALL 更新 `docs/数据文件说明.md`，包含以下内容：
1. 数据文件总览表（路径、大小、用途、来源、是否必须）
2. 压缩包清单（zip 文件名、解压目标目录、使用方法）
3. 小型数据文件清单（保留原样的 .txt 文件）
4. 缺失数据文件说明（代码引用但仓库中不存在的文件）
5. 已内嵌数据的替代方案说明（spamTestSklearn、testDigitsSklearn）
6. 不使用 Git LFS 的数据管理方案

#### Scenario: 用户查阅文档
- **WHEN** 用户打开 `docs/数据文件说明.md`
- **THEN** 能清晰了解每个数据文件的位置、用途和获取方式
- **AND** 能知道哪些 zip 文件需要解压、解压到何处
- **AND** 能知道哪些数据文件缺失、如何获取

## MODIFIED Requirements

### Requirement: 数据文件管理方式
原方案使用 Git LFS 管理大文件，现改为 zip 压缩包 + 手动管理方式。不再需要安装 Git LFS。
