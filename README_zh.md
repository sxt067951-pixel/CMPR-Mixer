# CMPR-Mixer

[English](README.md)

论文《基于情境引导多尺度模式检索的电力负荷预测》的实现代码。

## 研究概述

CMPR-Mixer 利用过去 168 小时的负荷及外生信息，预测未来 24 小时的用电负荷。两个时段的平均负荷可能相近，峰值却有明显差异。相似的历史负荷曲线在高温天气或节假日也可能呈现不同的后续走势。为在降采样时保留局部峰值信息，模型根据窗口内最大值与均值之差，对平均负荷进行可学习的修正。随后，模型结合近期负荷、气象和日历条件，从可学习的模式字典中检索相关模式，用于融合不同时间尺度的季节性波动和长期趋势。训练时，模型同时约束逐小时预测误差和频域误差，以更好地刻画未来 24 小时负荷曲线的整体形态。

实验采用中国广西 10 个城市、369 台变压器的小时负荷记录，并按城市汇总。分别在各城市训练和测试时，CMPR-Mixer 的十城平均 MSE 为 **0.1454**，MAE 为 **0.2711**，两项误差均为所比较模型中的最低值。为检验学到的负荷模式能否迁移到新城市，我们在九个城市上训练模型，冻结参数后在第十个城市上测试。依次留出每个城市后，十城平均 MSE 和 MAE 分别为 **0.1332** 和 **0.2505**。上述指标均基于标准化负荷计算，各城市在平均值中权重相同。

![CMPR-Mixer 总体架构，展示多尺度负荷分解、情境引导模式检索以及季节项与趋势项的跨尺度融合][cmpr-architecture]

*论文图 5。季节分支从细尺度向粗尺度传递信息，趋势分支从粗尺度向细尺度传递信息。*

## 环境安装

在仓库根目录执行以下命令，创建 Python 环境并安装指定版本的依赖。

```bash
conda create -n cmpr-mixer python=3.9 -y
conda activate cmpr-mixer
python -m pip install -r requirements.txt
```

## 数据准备

从 [Figshare 数据集](https://doi.org/10.6084/m9.figshare.26333452.v1) 下载 **Transformer_DB.db**，放入 `data/raw/`，再生成小时级 CSV 文件。

```bash
python prepare_data.py --db_path data/raw/Transformer_DB.db --output_dir data/transformer
```

训练使用的文件包括十个城市的 `transformer_load_city_0_exog.csv` 至 `transformer_load_city_9_exog.csv`，以及总负荷文件 `transformer_load_all_exog.csv`。每行依次为时间戳、56 个按固定顺序排列的外生变量和目标负荷。已有预处理文件时，可通过 `--data_root` 指定其目录。[字段说明](data/README.md) · [数据集论文](https://doi.org/10.1038/s41597-025-04766-7)

## 训练与测试

首次运行可从单城市开始。以下命令训练城市 0 的模型，根据验证集负荷 MSE 选择最佳权重，随后在测试集评估并保存预测值。

```bash
python run.py --cities 0 --skip_total --save_predictions
```

运行论文中用于计算十城平均误差的十个城市内预测实验，可执行以下命令。去掉 `--skip_total` 后会额外训练一个总负荷模型。

```bash
python run.py --all_cities --skip_total
```

数据按时间先后划分，训练集、验证集、测试集约占 70%、10%、20%。如需重新评估城市 0 已保存的模型，沿用原数据和模型设置，执行 `python run.py --cities 0 --skip_total --mode test`。使用 CPU 时添加 `--cpu`。复现论文实验时使用默认的 `--time_alignment legacy`，比较不同模型变体时应统一该设置。

## 消融与跨城市迁移

论文的消融表使用城市 0–4。以下命令在这五个城市上分别运行完整模型，以及移除输入门控、改用平均池化、移除模式检索、仅使用时域 MSE 的四个变体。

```bash
python run.py --cities 0 1 2 3 4 --skip_total --variants full no_gate avg_pool no_prototype mse_loss --output_root outputs/ablation
```

| 迁移实验 | 运行命令 |
| --- | --- |
| 九城训练、留一城测试，依次轮换十个目标城市 | `python transfer/city_loco_transfer.py` |
| 按历史负荷相关性分组，对城市对进行双向迁移 | `python transfer/city_similarity_transfer.py` |
| 城市 0–4 与城市 5–9 两组之间的双向迁移 | `python transfer/city_group_transfer.py --direction both` |

迁移实验需要备齐十个城市的 CSV。上述设置在目标城市测试时冻结迁入模型的参数，各城市分别使用自身训练时段的数据计算标准化统计量。

如需比较不同目标城市训练样本比例下的微调效果，执行以下命令。

```bash
python transfer/city_finetune_transfer.py --ratios 0 5 10 20
```

## 输出结果

城市内预测的汇总 CSV 位于 `outputs/forecast/`，每次运行的子目录保存模型权重 `checkpoint.pth` 和评价结果 `metrics.json`。论文使用标准化负荷的 `mse` 和 `mae`，`raw_*` 字段对应还原到原始单位后计算的误差。添加 `--save_predictions` 后，子目录还会保存预测值 `predictions.npy` 和真实值 `targets.npy`。迁移实验的汇总结果位于 `outputs/transfer/`。

## 致谢

CMPR-Mixer 的多尺度结构基于 TimeMixer，原型检索的实现参考了 ProtoTS。[第三方代码说明](NOTICE)

[cmpr-architecture]: figures/CMPR-Mixer-architecture.png
