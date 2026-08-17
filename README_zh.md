# WTI区间预测的多源数据与特征预处理

本仓库用于公开论文中多源输入数据及最终特征矩阵的构建材料。

## 公开范围

仓库仅支持复现**数据构建与特征预处理阶段**，包括：

- 研究使用的原始数据文件；
- 经过时间对齐的最终特征矩阵；
- 用于时间对齐、Spearman相关性筛选、PCA、PLS以及BERT情感得分日度聚合的Python代码和Jupyter Notebook；
- 变量说明、数据来源说明和文件校验值。

仓库不包含预测模型实现、CEEMDAN分解、图相似节点构建、ELM/GRU/KAN-LSTM训练、组合预测和预测结果文件。模型结构、样本划分、超参数、重复实验设置及评价指标已在论文中说明。

## 目录结构

```text
data/raw/                         预处理使用的原始数据
data/processed/                   最终八变量模型输入矩阵
notebooks/feature_preprocessing.ipynb
preprocessing/feature_preprocessing.py
data_dictionary.md
MANIFEST.sha256
requirements.txt
```

## 处理后矩阵

`data/processed/WTI_interval_feature_matrix.xlsx`包含2020年1月2日至2025年9月15日的1,469个日度样本，变量为：

`Date、UB、LB、BI1、BI2、NH、SD1、SD2`。

该文件不包含相似节点、分解分量、模型预测值或评价结果。

## 运行方法

建议使用Python 3.10。在仓库根目录运行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python preprocessing\feature_preprocessing.py
```

运行结果保存在`preprocessing_outputs/`，其中最终矩阵为`09_final_multisource_feature_matrix.xlsx`。也可以从仓库根目录打开`notebooks/feature_preprocessing.ipynb`并按顺序运行全部单元格。

## 复现验证

本仓库的预处理流程已完成实际运行验证，可筛选出与原实验一致的11个百度指数变量，并生成1,469行、8列的最终矩阵。`UB、LB、NH、SD1、SD2`与公开矩阵完全一致，`BI1、BI2`的最大绝对差异低于`3.1e-11`，仅为浮点计算精度差异。

## 数据来源与使用

原始数据提供方、来源网站、获取期间及收集方法见论文正文。`news_text_with_sentiment_scores.xlsx`提供清洗后的新闻文本和预先计算的BERT情感得分；本仓库对该得分进行日度聚合，但不重新训练BERT模型。

本仓库不对第三方来源数据另行主张再分发许可，数据再利用应遵守原始提供方的使用条款。

