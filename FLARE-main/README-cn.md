# FLARE：恒星耀斑预测框架

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

本仓库为IJCAI-25论文《FLARE：基于恒星物理属性与历史记录的耀斑预测框架》官方代码库。论文地址：https://arxiv.org/abs/2502.18218

## 📖 简介

恒星耀斑是天文学研究的重要观测样本，但已记录的耀斑事件数量有限。耀斑预测可提供更多事件样本来辅助研究，但目前尚无专用模型。本文通过大量实验证明，恒星物理属性和历史耀斑记录对预测任务均有价值。我们由此提出**FLARE**（基于特征融合的光变曲线天文记录预测框架），这是首个专用于恒星耀斑预测的大模型。FLARE通过**软提示模块**和**残差记录融合模块**整合两类输入。在公开Kepler光变曲线数据集上的实验表明，FLARE在所有评估指标上均优于对比方法。最后，我们通过详细案例验证了模型的预测能力。

## 🚀 模型架构

![](figures/flare_framework.jpg)

* **软提示模块**：将表格形式的恒星物理属性及其名称转换为文本描述，利用P-tuning区分不同恒星，实现星体特异性特征检测。
* **残差记录融合模块**：将光变曲线分解为趋势项和残差项，将历史耀斑记录与残差融合，增强模型鲁棒性。
* **PLM微调**：采用LoRA等参数高效微调方法，冻结预训练大模型（PLM）的大部分参数，使其适应跨模态输入（物理属性文本和光变曲线）。

## 🛠️ 安装

克隆仓库并安装依赖：

```bash
git clone https://github.com/CASIA-LMC-Lab/FLARE.git
cd FLARE
pip install -r requirements.txt
```

## 📂 数据集准备

实验使用Kepler光变曲线数据集，包含以下文件：
* `kepler_flare_id_start_end.npy`
* `kepler_meta_feats3.npy`
* `kids.npy`
* `seqlen_512_horizon_48_stride_50_noPeriod.tar.gz`
* `Kepler_npy.tar.gz`

请从[Google Drive](https://drive.google.com/drive/folders/1vzjyoLW8UjiaUQAj9UegtCU9p_Jf6GWm?usp=drive_link)下载数据集。下载解压后，修改`Configs.py`中的数据集路径（如`<DATASET_ROOT>`和`<PROJECT_ROOT>`）以匹配本地目录结构。

## 💻 运行与训练

数据与环境就绪后，执行以下命令开始训练：

```bash
python LC_classification.py \
    --llm_model BERT \
    --tslm_model BERT \
    --n_runs 3 \
    --flare_process_type FE \
    --id_process_type uniq \
    --ifP_tuning \
    --ifLoRA 
```

**主要参数说明：**
* `--llm_model`：用于提取文本特征的大语言模型（如`BERT`、`GPT2`、`Llama`）。
* `--tslm_model`：时间序列处理模型。
* `--n_runs`：实验重复次数，用于获取平均性能指标。
* `--flare_process_type`：耀斑特征处理方式（示例中设为`FE`）。
* `--ifP_tuning`：启用P-tuning处理软提示文本特征。
* `--ifLoRA`：启用LoRA对骨干PLM进行高效微调。

## 📚 引用

如本代码或工作对您的研究有帮助，请引用：

```bibtex
@inproceedings{zhu2025flare,
  title={FLARE: A Framework for Stellar Flare Forecasting Using Stellar Physical Properties and Historical Records},
  author={Zhu, Bingke and Wang, Xiaoxiao and Jia, Minghui and Tao, Yihan and Kong, Xiao and Luo, Ali and Chen, Yingying and Tang, Ming and Wang, Jinqiao},
  booktitle={Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25)},
  year={2025}
}
```

## 📜 许可证

本项目采用MIT许可证。版权所有 © 2026 CASIA-LMC-Lab。完整许可条款请参见根目录下的`LICENSE`文件。