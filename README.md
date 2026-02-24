# QiChat

**QiChat** 是一个专为对话场景设计的 Decoder-only 语言模型，旨在提供性能可控、指标稳定、易于扩展的通用对话解决方案。
该项目涵盖从 **模型定义、预训练、监督微调（SFT）、推理和评估的完整落地流程实现** ，便于快速部署和学习。

“Qi”（祁）的命名灵感来源于我第一次肉眼看到银河的地方——祁连山。广阔、深邃和神秘正需要不断地探索和学习。
该项目隶属于我个人的项目集合 **QiAlgo**，因此命名为 **QiChat**。

---

## 🚀 安装

建议使用 Python ≥ 3.12，并提前安装 PyTorch（含 CUDA 支持）。
训练时如需使用 FlashAttention，请根据您的平台单独编译或安装相应 wheel。

```bash
pip install -r requirements.txt
```

---

## 🛠️ 使用

项目主要入口均位于 `scripts` 目录下，首次执行训练框架会创建所需的配置文件 `config.json` 并主动退出，请根据需要修改配置文件。

数据集文件请放置在 `data/dataset/` 目录下，仅支持 `.parquet` 格式，训练时会加载该目录以及递归子目录下的所有数据文件。

分词器模型请放置在 `data/tokenizer/model/` 目录下，支持 HuggingFace TokenizerFast 格式。（推荐使用本人中、英文分词器项目 [QiTianTokenizer](https://huggingface.co/Morton-Li/QiTianTokenizer-Base)，可选 12k～128k 词表大小）。

### 进行训练

`nprocs` 参数指定 DDP 训练使用的 GPU 数量，不提供时默认为 1（不使用 DDP）。

```bash
python scripts/pretrain.py [--nprocs N]
```

### 进行监督微调 (SFT)

```bash
python scripts/finetuner.py [--nprocs N]
```

---

## 📄 许可证

本项目采用 **Apache 2.0** 许可。
