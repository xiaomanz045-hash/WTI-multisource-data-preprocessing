# WTI区间预测的多源数据预处理

本仓库公开论文中多源输入数据的原始数据、处理后特征矩阵及预处理代码。

## 公开范围

仓库支持复现以下步骤：

1. 原始数据的日期清洗与时间对齐；
2. Spearman相关性计算和变量筛选；
3. 对筛选后的百度指数变量进行未标准化的两成分PCA降维；
4. 对结构化变量进行两成分PLS降维；
5. 使用指定的中文财经BERT模型计算文章层面的新闻情感得分；
6. 情感得分的日度聚合、缺失值处理和多源特征整合；
7. 生成最终可直接用于后续实验的特征矩阵。


## 目录结构

```text
data/raw/                              原始数据
data/processed/                        时间对齐后的最终特征矩阵
notebooks/feature_preprocessing.ipynb  特征预处理Notebook
notebooks/bert_sentiment_analysis.ipynb
preprocessing/feature_preprocessing.py
preprocessing/bert_sentiment_analysis.py
data_dictionary.md
model_metadata.json
MANIFEST.sha256
requirements.txt
```

仓库不重新发布BERT模型权重。代码固定调用[`hw2942/bert-base-chinese-finetuning-financial-news-sentiment`](https://huggingface.co/hw2942/bert-base-chinese-finetuning-financial-news-sentiment)的版本`596188a9c884118e13984140a8b568a2252e01c2`，也支持传入本地模型目录。模型来源和标签映射记录在`model_metadata.json`中。

## 最终特征矩阵

`data/processed/WTI_interval_feature_matrix.xlsx`包括`Date`、`UB`、`LB`、`BI1`、`BI2`、`NH`、`SD1`和`SD2`。

- `BI1`和`BI2`为百度指数变量经过筛选和PCA降维后得到的两个成分；
- `NH`为BERT生成并按日聚合的新闻情感得分；
- `SD1`和`SD2`为结构化变量经过PLS降维后得到的两个成分。

研究者既可以直接使用该矩阵，也可以利用仓库中的原始数据和预处理代码重新生成。

## 安装

建议使用Python 3.10。在仓库根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 运行方法

根据清洗后的新闻文本重新计算文章层面的BERT情感得分：

```powershell
python preprocessing\bert_sentiment_analysis.py
```

模型输出`Negative`、`Neutral`和`Positive`，分别映射为`-1`、`0`和`1`。最大文本长度为512。若使用本地模型，可增加`--model-path <本地模型目录>`。

从原始数据重新生成最终多源特征矩阵：

```powershell
python preprocessing\feature_preprocessing.py
```

最终矩阵保存为`preprocessing_outputs/09_final_multisource_feature_matrix.xlsx`。同一次运行还会输出相关性表、筛选后的百度指数变量、PCA结果、日度情感得分、结构化变量对齐结果和PLS结果。

## 验证情况

预处理流程已实际运行通过，并筛选出与原实验一致的11个百度指数变量。重新生成的`UB`、`LB`、`NH`、`SD1`和`SD2`与原实验完全一致；`BI1`和`BI2`的最大绝对差异低于`3.1e-11`，仅为浮点计算误差。

## 数据来源与使用

原始数据提供方、来源网站、获取期间和收集方法见论文正文。`news_text_with_sentiment_scores.xlsx`包含清洗后的新闻文本和文章层面的BERT情感得分。仓库提供重新生成情感得分的推理代码，但不对第三方BERT模型进行微调或重新训练。

本仓库不对第三方来源数据另行主张再分发许可，数据再利用应遵守原始提供方的使用条款。
