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

---

### 术语和定义

- **快速评估**：在训练过程中使用当下模型的权重，对少量且固定的验证集进行评估。
- **训练步**：计算设备接受的最小训练单位，也称 `min-step`，每步会处理一个批次的数据。
- **优化步**：模型权重更新步，也称 `opt-step`（优化器步），计算公式为 `min-step x grad_accum_steps`，故：当 grad_accum_steps 为 1 时训练步 = 优化步。

---

### ⚙️配置文件解析

配置文件 `config.json` 包含了训练器的所有超参数设置：

- `device`: 训练设备选择，支持 auto（自动选择）、cpu、cuda、mps，当选择 auto 时会根据系统环境优先使用更快的硬件，当使用 DDP 时必须指定为 auto 或 cuda。
- `log_level`: 日志记录级别，支持 DEBUG、INFO、WARNING、ERROR、CRITICAL，低于设置级别的日志将被忽略。
- `model`
  - `param_size`: 模型参数量级，支持 Tiny (功能测试和调试用途，仅 2M 参数)、73M、0.3B、0.6B、1.3B、3.2B。
  - `attn_implementation`: 注意力机制实现，支持 flash_attention_2（推荐，需安装 FlashAttention 2）、flash_attention_3（硬件提供完整支持时使用）、sdpa、eager。
  - `dtype`: 模型权重精度类型，会自动配置并应用精度调整，在硬件提供完整支持时推荐 bfloat16。
  - `enable_grad_checkpointing`: 是否启用梯度检查点，除非你知道在做什么，否则不建议启用。
  - `model_config_kwargs`: 其他模型配置参数，会以最高优先级透传给模型配置类。
- `training`
  - `batch_size`: 批次大小，在使用 DDP 时指每个设备的批次大小。
  - `learning_rate`: 学习率（峰值）。
  - `min_learning_rate`: 最小学习率，学习率调度器会将学习率从 `learning_rate` 线性衰减到 `min_learning_rate` 后停止衰减。
  - `num_epochs`: 训练的总轮数。
  - `warmup_ratio`: 学习率预热占 **单轮次** **优化步** 数的比率。假设每轮次有 1000 个优化步，warmup_ratio 为 0.1，则预热阶段将从第 0 步学习率从 0 经过 100 步线性增加到 `learning_rate`。
  - `gradient_clipping_max_norm`: 梯度裁剪的最大范数，
  - `quick_eval_per_epoch`: 在每个 epoch 中进行多少次**快速评估**，快速评估指标有所改善时会保存一个检查点。
  - `freeze_layers`: 冻结前 n 层参数不进行训练，设置为 0 则不冻结。
  - `freeze_embeddings`: 是否冻结词嵌入层参数不进行训练。
  - `grad_accum_steps`: 梯度累积步数，每 n 训练步进行一次梯度更新。
- `seed`: 随机种子，设置为固定值以确保训练过程的可复现性。
- `dataset`
  - `max_input_length`: 最大输入长度，超过该长度的输入将被舍弃，设置为 -1 则不限制。
  - `test_split`: 从训练样本中划分出多少条样本作为测试集，取值 [0, 1) 表示比例，[1, ∞) 表示数量，小于 0 则不划分测试集。
  - `field_name`: 从数据集中哪个字段加载输入，该字段应已经分词器处理并转换为 token id，格式应为 numpy.array。
- `notifications`: 训练通知配置，目前仅支持 email（电子邮件），目前用处不多，主要是在训练完成时发送通知。
  - `SmtpChannel`：SMTP 电子邮件通知通道配置（目前唯一支持的通知通道）
    - `host`: SMTP 服务器地址。
    - `port`: SMTP 服务器端口。
    - `username`: SMTP 服务器用户名。
    - `password`: SMTP 服务器密码。
    - `to_addresses`: 接收通知的电子邮件地址列表，如: ["address_1@example.com", "address_2@example.com"]。
    - `use_tls`: 是否使用 TLS 加密连接到 SMTP 服务器, false 表示使用 ssl 连接。
- `max_checkpoint_count`: 检查点数量上限，超过该数量时会自动删除本次训练过程中最旧的检查点。

---

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
