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

> 由于语法差异，明确不兼容 Python 3.11 及以下版本，强“兼”难度不大，欢迎提交 PR 添加对 Python 3.11 的支持，但本人更建议向前看。

---

## 术语和定义

- **快速评估**：在训练过程中使用当下模型的权重，对少量且固定的验证集进行评估。
- **训练步**：计算设备接受的最小训练单位，也称 `micro-step`，每步会处理一个批次的数据。
- **优化步**：模型权重更新步，也称 `opt-step`（优化器步），计算公式为 `micro-step x grad_accum_steps`，故：当 grad_accum_steps 为 1 时训练步 = 优化步。
- **DDP**：分布式数据并行（Distributed Data Parallel）。
- **检查点**：训练过程中保存的模型权重和训练状态的快照，通常在训练过程中定期保存，以便在需要时恢复训练或进行评估。
- **Unlikelihood**：非似然，简称 `UL`，一种辅助损失的实现，用于弥补最大化似然(MLE)训练的缺陷，提高生成模型的多样性并降低重复率，详见[损失函数](#损失函数)部分。

---

## 🛠️ 使用

项目主要入口均位于 `scripts` 目录下，首次执行训练框架会创建所需的配置文件 `config.json` 并主动退出，请根据需要修改配置文件，详见[配置文件解析](#配置文件解析)。

### 前置条件

- 数据集: 仅支持 `.parquet` 格式，请放置在 `data/dataset/` 目录下，详见[数据集](#数据集)部分。
- 分词器: 分词器模型请放置在 `data/tokenizer/model/` 目录下，详见[分词器](#分词器)部分。

---

### 进行训练

训练脚本均已配置 she-bang，确保有可执行权限后可直接执行，会自动使用环境变量中的 python，更建议使用 python 执行以确保使用正确的 Python 环境。

本框架使用 TensorBoard 进行训练日志记录，会自动生成并将日志记录在在 `logs/` 目录下，建议使用 TensorBoard 可视化工具进行查看和分析。

参数：
- `nprocs`: 指定 DDP 使用的 GPU 数量，不提供时默认为 1（不使用 DDP）。

#### 进行预训练 (PreTrain)
```bash
python scripts/pretrain.py [--nprocs N]
```

#### 进行监督微调 (SFT)

```bash
python scripts/finetuner.py [--nprocs N]
```

#### 使用本框架对 Qwen3 模型进行预训练

```bash
python scripts/qwen3_pretrain.py [--nprocs N]
```

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
  - `optimizer_beta`: AdamW 优化器 beta 设置。
  - `num_epochs`: 训练的总轮数。
  - `warmup_ratio`: 学习率预热占 **单轮次** **优化步** 数的比率。假设每轮次有 1000 个优化步，warmup_ratio 为 0.1，则预热阶段将从第 0 步学习率从 0 经过 100 步线性增加到 `learning_rate`。
  - `gradient_clipping_max_norm`: 梯度裁剪的最大范数，
  - `quick_eval_per_epoch`: 在每个 epoch 中进行多少次**快速评估**，快速评估指标有所改善时会保存一个检查点。
  - `freeze_layers`: 冻结前 n 层参数不进行训练，设置为 0 则不冻结。
  - `freeze_embeddings`: 是否冻结词嵌入层参数不进行训练。
  - `grad_accum_steps`: 梯度累积步数，每 n 训练步进行一次梯度更新。
  - `auxiliary_loss`:
    - `weight`: 辅助损失权重，设置为 0 则不使用辅助损失，计算公式为 `loss = ce_loss + (auxiliary_loss_weight * total_auxiliary_loss)`。
    - `recent_unlikelihood`: recent-token Unlikelihood Loss 配置
      - `window_size`: Recent UL 的窗口大小，即 w 的值。
      - `max_neg_per_pos`: Recent UL 的惩罚范围 k 的最大值。
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

### 📊数据集

数据集文件应放置在 `data/dataset/` 目录下，训练时会加载该目录以及递归子目录下的所有 parquet 文件。
数据集应包含一个字段（默认为 `tokenized`，可通过配置文件 `dataset.field_name` 修改）存储输入文本的 token id 数组，格式为 numpy.array，可调整精度以节省存储空间，例如使用 uint16 代替 int32。

具体数据集组织方式和预处理方法由于版权和隐私问题仅提供部分示例，脚本位于 `scripts/dataset/` 目录下，保留了一些工具函数以便用户根据自己的情况借鉴和使用。

---

### 分词器

分词器模型应放置在 `data/tokenizer/model/` 目录下，支持 HuggingFace TokenizerFast 格式。
建议使用 [QiTianTokenizer](https://huggingface.co/Morton-Li/QiTianTokenizer-Base) 分词器项目，该分词器为通用分词器，提供了多种词表大小（12k～128k）以适应不同需求，且在中英文对话场景中表现良好。

在当前训练框架下，并未实现对其他分词器的适配，不过鉴于低耦合的设计原则，适配其他分词器是相对简单的工作，但目前没有计划实现这一功能。

---

### 训练器

本项目参考了 Qwen3 的数据处理方式，
简单来说就是未在数据集中使用 BOS（begin of sequence）标记，同时又不想因为这一变化设计一个功能开关，所以在训练器的预训练、监督微调训练中做了一个特殊处理，具体可在 inference_samples 函数中了解。

---

### 损失函数

#### recent-token Unlikelihood Loss

Recent-token Unlikelihood Loss 是基于原版 Unlikelihood Training（非似然训练）的一种改良实现。
根据试验，发现由于 MLE 目标下的 CE Loss 缺陷，导致即便模型在 top-k 指标还不错的情况下，在遇到吸引子后叠加自我强化（self-reinforcement），易导致生成文本的多样性不足和重复率过高问题。

Recent-token Unlikelihood Loss 的核心思想是：在训练过程中，除了最大化正确 token 的似然之外，还引入一个辅助损失，在 CE 抬高 gold token 的概率的同时，压低吸引子的输出欲望，可激励模型打断自我强化的循环，

经过试验，发现吸引子通常来自 t 之前的位置，意味着模型的吸引子并非完全由模型主动生成再吸引自己（尤其是教师强制训练下），且吸引子通常在 t 之前的 w 个时间步内，不会跨越更远的位置，
因此 recent-token Unlikelihood Loss 的核心改良在于：

- 无需手动设计吸引子列表，完全基于 gold token 进行计算，简化了实现和使用；
- 可调的窗口大小 w，以及惩罚范围 k，尽可能不影响 MLE 目标；
- 惩罚范围 k 采用“最近优先”原则，即在窗口内距离 t 越近的 token 越优先纳入惩罚范围，避免合理重复被抑制。

---

## References

- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [Neural Text Generation with Unlikelihood Training](https://arxiv.org/abs/1908.04319)

---

## Citation 引用

If you use this project in your research, please consider citing:

```bibtex
@misc{QiChat,
  title  = {QiChat: A General-Purpose Conversational Decoder-Only Language Model},
  author = {Morton Li},
  year   = {2026},
}
```

---

## 📄 License 许可

本项目采用 **Apache 2.0** 许可。
