from typing import Any, Callable, Literal

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle


class ModelProbe:
    def __init__(self):
        self._hooks: list[RemovableHandle] = []
        self._hidden_states_buffers: dict[str, torch.Tensor] = {}
        self._results_buffers: dict[str, dict[str, dict[str, int | float] | int | float]] = {
            'representation_diversity_metrics': {},
            'output_distribution_metrics': {},
        }

        self._active: bool = False

    def attach(self, model: nn.Module, module_names: list[str],) -> None:
        """
        Attach hooks to the specified modules in the model.
        Args:
            model (nn.Module): 要监视的模型
            module_names (list[str]): 要监视的模块名称列表，支持嵌套模块，例如 ['encoder.layer.0', 'encoder.layer.1']。
        """
        self.detach()
        for module_name in module_names:
            module = model.get_submodule(target=module_name)
            hook = module.register_forward_hook(self._module_hook(module_name))
            self._hooks.append(hook)

    def detach(self) -> None:
        """Remove all hooks and buffers."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._hidden_states_buffers.clear()
        self._results_buffers.clear()

    def set_active(self, active: bool) -> None:
        """Activate or deactivate the probe."""
        if not active:
            self._hidden_states_buffers.clear()
            self._results_buffers.clear()
        self._active = active

    @property
    def active(self) -> bool: return self._active
    @property
    def results(self) -> dict[str, int | float | dict[str, int | float | dict[str, int | float | dict[str, int | float]]]]: return self._results_buffers
    @property
    def flattened_results(self) -> dict[str, dict[str, int | float | dict[str, int | float]]]: return {
        key: self._flatten_dict(data=value)
        for key, value in self._results_buffers.items()
    }

    def _flatten_dict(self, data: dict, parent_key: str = "", sep: str = "/") -> dict[str, int | float]:
        items: dict[str, int | float | dict[str, int | float]] = {}
        for key, value in data.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else str(key)

            if isinstance(value, dict):
                if all(isinstance(sub_value, (int, float)) for sub_value in value.values()):
                    items[new_key] = value
                else:
                    items.update(self._flatten_dict(value, new_key, sep=sep))
            elif isinstance(value, (int, float)):
                items[new_key] = value

        return items

    def _module_hook(self, module_name: str) -> Callable[[nn.Module, tuple, Any], None]:
        """Create a forward hook function that captures the output of the specified module."""
        def hook(_module: nn.Module, _input: tuple, _output: Any) -> None:
            if not self._active: return  # 如果探针未激活
            if torch.is_tensor(_output) and _output.dim() == 3: self._hidden_states_buffers[module_name] = _output.detach()
            elif isinstance(_output, (tuple, list)):
                for i, item in enumerate(_output):
                    if torch.is_tensor(item) and item.dim() == 3:  # 只捕获形状为 (batch_size, seq_length, hidden_size) 的张量，避免捕获到其他可能的输出
                        self._hidden_states_buffers[module_name] = item.detach()
                        break
            # elif isinstance(_output, ?):
        return hook

    @torch.no_grad()
    def compute_output_distribution_metrics(
        self,
        logits: torch.Tensor,
        labels: torch.LongTensor,
        topk: int = 5,
        ignore_index: int = -100,
        shift_mod: Literal['internal', 'external'] = 'internal',
        max_tokens: int | None = None,
    ) -> dict[str, int | float | dict[str, int | float]]:
        """
        计算输出分布指标
        Args:
            logits (torch.Tensor): 模型输出的logits，形状为(batch_size, seq_length, vocab_size)
            labels (torch.LongTensor): 真实标签，形状为(batch_size, seq_length)
            topk (int): 计算Top-k相关指标时的k值，默认为5
            ignore_index (int): 在计算指标时要忽略的标签索引，默认为 -100
            shift_mod (str): 模型损失计算的 shift 模式，分为 '内部' 和 '外部' 两种：
                - 'internal'：模型的输入和标签完全对齐，在进行损失计算时通过继承 PreTrainedModel 类的 loss_function 函数自动进行 shift 处理。
                - 'external'：模型的输入和标签存在一个 token 的错位，在输入时 input_ids = input_ids[:, :-1]，labels = labels[:, 1:]。
            max_tokens (int | None): 可选的最大token数量，如果提供，则只计算随机采样max_tokens个token的指标
        """
        if shift_mod not in ('internal', 'external'): raise ValueError(f"Invalid shift_mod: {shift_mod}. Must be 'internal' or 'external'.")
        if shift_mod == 'internal':
            # 对于 'internal' 模式，logits 和 labels 完全对齐，需要在计算指标时进行 shift 处理(实现方式与 ForCausalLMLoss 一致)
            labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
            labels = labels[..., 1:].contiguous()
        # 对于 'external' 模式，logits 和 labels 存在一个 token 的错位，input_ids = input_ids[:, :-1]，labels = labels[:, 1:]
        # 所以直接计算 valid_mask 即可，无需再进行 shift 处理
        valid_mask = labels != ignore_index

        # 展平
        valid_mask_flat = valid_mask.reshape(-1)
        # 如果没有有效的标签，则返回 NaN 来表示指标无法计算
        if not valid_mask_flat.any(): return {
            'logits_entropy': float('nan'),
            'logits_std': float('nan'),
            'topk': {
                'top_1_top2_confidence_gap': float('nan'),
                'top1_acc': float('nan'),
                'top5_acc': float('nan'),
                f'top{topk}_mass': float('nan'),
                'p_true_arith_mean': float('nan'),
            }
        }
        logits_flat = logits.reshape(-1, logits.size(-1))[valid_mask_flat]
        labels_flat = labels.reshape(-1)[valid_mask_flat]

        if max_tokens is not None and logits_flat.size(0) > max_tokens:
            targets_pos = torch.randperm(logits_flat.size(0), device=logits_flat.device)[:max_tokens]
            logits_flat = logits_flat[targets_pos]
            labels_flat = labels_flat[targets_pos]
        logits_flat = logits_flat.float()  # 数值稳定：转 fp32

        probs = nn.functional.softmax(logits_flat, dim=-1)
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1).mean().item()  # 平均熵

        logits_std = logits_flat.float().std(dim=-1).mean().item()  # logit 标准差，反映分布的“平坦程度”，越小表示越平坦

        p_true_arith_mean = probs.gather(1, labels_flat.unsqueeze(1)).mean().item()  # 真实标签的平均概率

        k_eval = min(max(topk, 5), probs.size(-1))
        topk_vals, topk_idx = torch.topk(probs, k=k_eval, dim=-1)

        confidence_gap = (topk_vals[:, 0] - topk_vals[:, 1]).mean().item()  # Top-1 和 Top-2 平均置信度差距

        top1_acc = (topk_idx[:, 0] == labels_flat).float().mean().item()  # Top-1准确率
        top5_acc = (topk_idx[:, :5] == labels_flat.unsqueeze(-1)).any(dim=-1).float().mean().item()  # Top-5准确率

        topk_eff = min(topk, topk_vals.size(1))  # 实际计算的 topk 值，避免超过 vocab_size 否则会计算出错误的指标
        topk_mass = topk_vals[:, :topk_eff].sum(dim=-1).mean().item()  # Top-k概率质量

        results = {
            'logits_entropy': entropy,
            'logits_std': logits_std,
            'topk': {
                'top_1_top2_confidence_gap': confidence_gap,
                'top1_acc': top1_acc,
                'top5_acc': top5_acc,
                f'top{topk}_mass': topk_mass,
                'p_true_arith_mean': p_true_arith_mean,
            },
        }
        self._results_buffers['output_distribution_metrics'] = results
        return results

    @torch.no_grad()
    def compute_representation_diversity_metrics(
        self,
        attention_mask: torch.BoolTensor | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, dict[str, int | float | dict[str, int | float]]]:
        """
        计算表征多样性指标
        同一样本内的 token-token 平均余弦相似度（越高越“塌缩”）
        Args:
            attention_mask (torch.BoolTensor | None): 输入的 attention mask，形状为(batch_size, seq_length)，其中 True 表示有效 token，False 表示 padding
            max_tokens (int | None): 可选的最大token数量，如果提供，则只计算随机 max_tokens个token 个指标
        """
        result: dict[str, dict[str, int | float | dict[str, int | float]]] = {
            'participation_ratio': {},
            'mean_cosine_similarity': {
                'intra': {},  # 同一样本内的 token-token 平均余弦相似度（越高越“塌缩”）
                'inter': {},  # 跨样本的 token-token 平均余弦相似度（越高越“塌缩”）
            }
        }
        use_attn_mask = attention_mask is not None and torch.is_tensor(attention_mask)
        if use_attn_mask and attention_mask.dtype != torch.bool:
            raise ValueError(f"Expected attention_mask to be a boolean tensor, but got {attention_mask.dtype}")

        for layer_name, hidden_states in self._hidden_states_buffers.items():
            # hidden_states 的形状是 (batch_size, seq_length, hidden_size)
            batch_size, seq_length, hidden_size = hidden_states.size()
            hidden_states = hidden_states.float()  # 数值稳定：转 fp32
            if use_attn_mask:
                mask = attention_mask  # 防止该函数在后续步骤错误调整 attention_mask，在同设备时为引用不影响性能，跨设备时不影响外部 attention_mask
                if mask.device != hidden_states.device: mask = mask.to(hidden_states.device)
                hidden_states_flat = hidden_states[mask]  # (batch_size * seq_length, hidden_size)
            else: hidden_states_flat = hidden_states.reshape(-1, hidden_size)  # 展平为 (batch_size * seq_length, hidden_size)
            if hidden_states_flat.numel() == 0:
                result['mean_cosine_similarity']['intra'][layer_name] = float('nan')
                result['mean_cosine_similarity']['inter'][layer_name] = float('nan')
                result['participation_ratio'][layer_name] = float('nan')
                continue

            intra_mean_cosine_similarity_values: list[torch.Tensor] = []
            # 计算单样本采样数量
            num_tokens_per_sample = max(max_tokens // batch_size, 2) if max_tokens is not None else seq_length
            for i in range(batch_size):
                if use_attn_mask:
                    mask_i = mask[i]  # (seq_length,)
                    hidden_states_i = hidden_states[i][mask_i]  # (num_tokens_i, hidden_size)
                else: hidden_states_i = hidden_states[i]  # (seq_length, hidden_size)
                if hidden_states_i.numel() == 0 or hidden_states_i.size(0) < 2: continue  # 如果该样本没有有效 token 或者 token 数量不足以计算余弦相似度，则跳过

                i_sample_pos = torch.randperm(hidden_states_i.size(0), device=hidden_states_i.device)[:num_tokens_per_sample]
                hidden_states_i = hidden_states_i[i_sample_pos]  # (num_tokens_per_sample, hidden_size)
                hidden_states_i_normed = nn.functional.normalize(hidden_states_i, dim=-1)  # (num_tokens_i, hidden_size)

                i_sample_pos1 = torch.randperm(hidden_states_i_normed.size(0), device=hidden_states_i_normed.device)
                sample1 = hidden_states_i_normed[i_sample_pos1]  # (num_tokens_per_sample, hidden_size)
                i_sample_pos2 = torch.randperm(hidden_states_i_normed.size(0), device=hidden_states_i_normed.device)
                sample2 = hidden_states_i_normed[i_sample_pos2]  # (num_tokens_per_sample, hidden_size)
                cosine_sim = (sample1 * sample2).sum(dim=-1).mean()  # 同一样本内的 token-token 平均余弦相似度
                intra_mean_cosine_similarity_values.append(cosine_sim)
            if intra_mean_cosine_similarity_values: result['mean_cosine_similarity']['intra'][layer_name] = torch.stack(intra_mean_cosine_similarity_values).mean().item()  # 同一样本内的 token-token 平均余弦相似度（越高越“塌缩”）
            else: result['mean_cosine_similarity']['intra'][layer_name] = float('nan')

            # 当前 hidden_states_flat 形状为 (num_tokens, hidden_size)
            # 其中 num_tokens 是有效 token 的总数，
            # 当 attention_mask 提供时，num_tokens = sum(attention_mask)，否则 num_tokens = batch_size * seq_length
            max_tokens_eff = min(hidden_states_flat.size(0), max_tokens) if max_tokens is not None else hidden_states_flat.size(0)
            sample_pos = torch.randperm(hidden_states_flat.size(0), device=hidden_states_flat.device)[:max_tokens_eff]
            hidden_states_flat = hidden_states_flat[sample_pos]  # (max_tokens_eff, hidden_size)

            # 跨样本 token 平均余弦相似度
            hidden_states_flat_normed = nn.functional.normalize(hidden_states_flat, dim=-1)  # (max_tokens_eff, hidden_size)
            # 随机采样两份 token 表征做近似，避免 O(N^2)
            sample_pos1 = torch.randperm(hidden_states_flat_normed.size(0), device=hidden_states_flat_normed.device)
            sample1 = hidden_states_flat_normed[sample_pos1]  # (max_tokens_eff, hidden_size)
            sample_pos2 = torch.randperm(hidden_states_flat_normed.size(0), device=hidden_states_flat_normed.device)
            sample2 = hidden_states_flat_normed[sample_pos2]  # (max_tokens_eff, hidden_size)
            cosine_sim = (sample1 * sample2).sum(dim=-1).mean().item()  # 跨样本平均余弦相似度
            result['mean_cosine_similarity']['inter'][layer_name] = cosine_sim

            # 有效维度（participation ratio）：对 centered features 做协方差谱近似
            # PR = (sum λ)^2 / sum(λ^2)，越小表示越“挤在少数方向”
            hidden_states_flat_centered = hidden_states_flat - hidden_states_flat.mean(dim=0, keepdim=True)
            cov = (hidden_states_flat_centered @ hidden_states_flat_centered.t()) / max(1, hidden_states_flat_centered.size(0) - 1)  # 协方差矩阵，形状 (sample_size, sample_size)
            eigenvalues = torch.linalg.eigvalsh(cov).clamp_min(0)  # 协方差矩阵的特征值，理论上应该非负，但数值误差可能导致小的负值，clamp_min(0) 来修正
            pr = ((eigenvalues.sum() ** 2) / (eigenvalues.square().sum().clamp_min(1e-12))).item()  # 参与度比，越小表示越“挤在少数方向”
            result['participation_ratio'][layer_name] = pr

        self._results_buffers['representation_diversity_metrics'] = result
        return result
