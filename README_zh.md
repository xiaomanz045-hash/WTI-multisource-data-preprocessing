# WTI区间预测的多源特征构建

本仓库公开论文中完整的特征数据构建材料。

## 公开范围

仓库支持复现以下步骤：

1. 原始数据的时间对齐；
2. Spearman相关性筛选；
3. 百度指数变量的PCA降维；
4. 结构化变量的PLS降维；
5. 使用指定中文财经BERT模型计算新闻情感得分并进行日度聚合；
6. 对全部特征值和目标值进行CEEMDAN分解；
7. 计算样本熵并重构低频、中频和高频序列；
8. 使用2V-GCN计算每个时点最相似的前2个历史区间节点。



## 目录结构

```text
data/raw/                              原始数据
data/processed/                        时间对齐后的八变量特征矩阵
artifacts/frozen_ceemdan.xlsx          原实验CEEMDAN冻结结果
artifacts/frozen_similarity/           原实验三频相似节点冻结结果
notebooks/feature_preprocessing.ipynb  数据预处理Notebook
notebooks/bert_sentiment_analysis.ipynb
preprocessing/bert_sentiment_analysis.py
preprocessing/feature_preprocessing.py
preprocessing/ceemdan_reconstruction.py
preprocessing/two_view_gcn_similarity.py
preprocessing/run_feature_pipeline.py
data_dictionary.md
MANIFEST.sha256
requirements.txt
```

仓库不重新发布BERT模型权重。代码固定调用官方模型[`hw2942/bert-base-chinese-finetuning-financial-news-sentiment`](https://huggingface.co/hw2942/bert-base-chinese-finetuning-financial-news-sentiment)及版本提交`596188a9c884118e13984140a8b568a2252e01c2`，也支持传入本地模型目录。模型来源和标签映射记录在`model_metadata.json`中。

## 运行方法

建议使用Python 3.10。在仓库根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

根据清洗后的新闻文本重新计算文章层面的BERT情感得分：

```powershell
python preprocessing\bert_sentiment_analysis.py
```

模型输出`Negative、Neutral、Positive`三类，分别映射为`-1、0、1`，最大文本长度为512。若使用本地模型，可增加`--model-path <本地模型目录>`。结果保存在`sentiment_outputs/news_text_scored_by_bert.xlsx`。

从原始数据重新生成多源特征矩阵：

```powershell
python preprocessing\feature_preprocessing.py
```

直接使用原实验冻结的CEEMDAN和相似节点结果，复现论文实际使用的三频特征输入：

```powershell
python preprocessing\run_feature_pipeline.py --mode frozen
```

重新运行CEEMDAN和2V-GCN：

```powershell
python preprocessing\run_feature_pipeline.py --mode recompute
```

也可以把刚生成的预处理矩阵直接传入后续流程：

```powershell
python preprocessing\run_feature_pipeline.py --mode recompute --feature-matrix preprocessing_outputs\09_final_multisource_feature_matrix.xlsx
```

生成的三频数据及相似节点输入保存在`feature_outputs/`。流程在预测模型训练之前结束。

## 随机性说明

原实验运行CEEMDAN和2V-GCN时没有保存随机种子。因此，`--mode frozen`用于精确复现论文实际采用的历史特征输入；`--mode recompute`保持原算法和超参数不变，但随机实现的具体数值可能与历史运行存在差异。代码提供可选的`--seed`参数，便于今后的确定性实验，但该参数没有追溯性地用于原实验。

保留的主要设置包括：CEEMDAN的`Nstd=0.2`、`NR=100`和`MaxIter=10`；样本熵的`m=2`和`r=0.2×标准差`；最相似历史节点数为2；可视图阈值为`0.5`；以及原三层GCN训练设置。

## 验证情况

原始数据预处理流程已实际运行通过，筛选出与原实验一致的11个百度指数变量，并生成1,469行、8列的最终矩阵。`UB、LB、NH、SD1、SD2`完全一致，`BI1、BI2`的最大绝对差异低于`3.1e-11`。冻结特征流程也已用于验证三类频率重构数据和前2个历史相似区间节点的拼接过程。

## 数据来源与使用

原始数据提供方、来源网站、获取期间和收集方法见论文正文。`news_text_with_sentiment_scores.xlsx`包含清洗后的新闻文本和文章层面的BERT情感得分。本仓库提供重新生成这些得分的推理代码，但不对第三方BERT模型进行微调或重新训练。

本仓库不对第三方来源数据另行主张再分发许可，数据再利用应遵守原始提供方的使用条款。
