# 新闻文本主题分类与 TF-IDF Softmax 优化

本项目实现了一个面向自然语言处理入门任务的新闻文本主题分类模块。项目不依赖外部语料库，而是构建合成短新闻数据，覆盖 technology、finance、sports、health 四类主题。相比简单关键词投票，TF-IDF Softmax 分类器能够学习词项权重，并在主题词交叉、公共词干扰的情况下给出更稳定的分类结果。

## 功能内容

- 构建 4 类新闻主题、共 380 条合成短文本样本。
- 为每类主题设置核心词表，并加入公共词、跨主题干扰词和随机新闻词。
- 实现 `keyword_baseline` 关键词投票基线方法。
- 实现词袋统计、TF-IDF 向量化和 `tfidf_softmax` 多分类模型。
- 使用 Numpy 实现 Softmax 交叉熵训练，不依赖深度学习框架。
- 统计准确率、Macro-F1 和混淆矩阵。
- 生成混淆矩阵、训练损失曲线、指标对比图和主题关键词权重图。
- 输出 `classification_predictions.csv` 和 `metrics.json`，方便复现实验和写提交说明。

## 运行方法

在本目录执行：

```bash
python main.py
```

也可以指定输出目录：

```bash
python main.py --output assets
```

运行后会生成：

- `confusion_matrix.png`：TF-IDF Softmax 分类器的混淆矩阵。
- `training_loss.png`：Softmax 训练损失曲线。
- `metric_comparison.png`：关键词基线与 TF-IDF Softmax 的指标对比。
- `top_words.png`：每个主题权重最高的关键词。
- `classification_predictions.csv`：测试集预测标签和置信度。
- `metrics.json`：核心指标和优化增量。

## 项目意义

该模块贴近神经网络课程中的文本表示和 Softmax 多分类。TF-IDF 可以看作传统文本特征工程，Softmax 分类器则是多分类神经网络的基础形式。项目展示了从文本分词、向量化、训练分类器到解释关键词权重的完整流程，也可以继续扩展到朴素贝叶斯、Word2Vec、TextCNN、RNN 或 Transformer 文本分类。

## 测试

```bash
python -m pytest tests
```

测试内容包括：

- TF-IDF Softmax 是否达到或超过关键词基线。
- 模型输出类别和混淆矩阵维度是否正确。
- 主程序是否能生成指标文件和全部可视化图片。
