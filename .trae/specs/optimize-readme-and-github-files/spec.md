# 优化 README 与 GitHub 公开文件 Spec

## Why
项目 README 缺乏专业外观（无居中标题、无徽章、无致谢），CONTRIBUTING.md 中文档链接指向旧编号，LICENSE 年份过时，缺少 CODE_OF_CONDUCT.md 等开源社区标准文件，需要全面优化以适配 GitHub 公开发布。

## What Changes
- 重写 README.md：居中标题、徽章、专业布局、致谢、链接有效性验证
- 更新 CONTRIBUTING.md：修正文档链接指向新编号 09-15
- 更新 LICENSE：修正年份
- 新建 CODE_OF_CONDUCT.md：开源社区行为准则
- 更新 requirements.txt：补充 lightgbm, catboost 依赖
- 更新 .gitattributes：移除不必要的 LFS 跟踪（项目不使用 LFS）
- **BREAKING** README.md 将从 `readme.md` 重命名为 `README.md`（GitHub 标准命名）

## Impact
- Affected docs: readme.md → README.md, CONTRIBUTING.md, LICENSE, requirements.txt, .gitattributes
- New files: CODE_OF_CONDUCT.md
- Affected code: 无源码变更

## ADDED Requirements

### Requirement: README 专业布局
README.md SHALL 包含以下结构，按顺序排列：
1. 居中项目标题 + 一句话描述
2. 徽章行（License、Python版本、测试状态）
3. 居中项目特色简介
4. 快速开始（安装+运行）
5. 学习路线（4阶段ASCII图）
6. 各阶段详解（表格）
7. 项目统计
8. 文档导航（折叠区）
9. 致谢（参考教材、开源项目、贡献者）
10. 开源协议与免责声明链接

### Requirement: 链接有效性
README.md 中所有链接 SHALL 指向实际存在的文件：
- docs/01-快速上手.md 至 docs/15-项目报告.md
- LICENSE, DISCLAIMER, CONTRIBUTING.md, CODE_OF_CONDUCT.md
- requirements.txt

### Requirement: 致谢章节
README.md SHALL 包含致谢章节，感谢以下参考来源：
- Han & Kamber《数据挖掘：概念与技术》
- Peter Harrington《Machine Learning in Action》
- scikit-learn、numpy、pandas、matplotlib 等开源项目
- 项目贡献者

### Requirement: CONTRIBUTING.md 链接修正
CONTRIBUTING.md 中所有文档链接 SHALL 使用新编号 09-15：
- docs/05-实施与开发文档.md → docs/13-实施与开发文档.md
- docs/02-需求规格说明书.md → docs/10-需求规格说明书.md

### Requirement: CODE_OF_CONDUCT.md
项目 SHALL 包含 CODE_OF_CONDUCT.md，采用 Contributor Covenant v2.1 中英双语版本。

### Requirement: requirements.txt 补充
requirements.txt SHALL 包含 lightgbm 和 catboost 依赖。

### Requirement: LICENSE 年份更新
LICENSE 中版权年份 SHALL 更新为 2024-2026。

## MODIFIED Requirements

### Requirement: README 文件名
README 文件名 SHALL 为 `README.md`（全大写），符合 GitHub 标准命名规范。原 `readme.md` 需重命名。

## REMOVED Requirements

### Requirement: .gitattributes LFS 跟踪
**Reason**: 项目不使用 Git LFS，所有大文件以 .zip 压缩形式管理。.gitattributes 中的 LFS 跟踪规则会导致未安装 LFS 的用户无法正常克隆。
**Migration**: 将 .gitattributes 清空或仅保留必要的文本规范化规则。
