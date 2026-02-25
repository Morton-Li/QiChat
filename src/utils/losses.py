from collections import deque
from typing import Optional

import torch


def compute_token_loss_weighting(
    labels: torch.Tensor,
    loss_weighting_map: dict[int, float],
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    计算指定 token 位置的损失加权张量。
    Args:
        labels (torch.Tensor): 真实标签张量，形状为 (batch_size, seq_length) 或 (batch_size * seq_length,)。
        loss_weighting_map (dict[int, float]): Mapping from token IDs to their corresponding loss weight multipliers. ({token_id: multiplier, ...})
        dtype (torch.dtype): Desired data type for the output tensor.
        device (torch.device): Desired device for the output tensor.
    Returns:
        torch.Tensor: 权重张量，形状与 labels 相同，在指定 token 位置应用了相应的权重，其他位置为 1.0。
    Example:
        labels = torch.tensor([[1, 2, 3], [4, 5, 6]])
        loss_weighting_map = {2: 0.5, 5: 2.0}
        weight_tensor = apply_token_loss_weighting(labels, loss_weighting_map, torch.float32, torch.device('cpu'))
        # weight_tensor 将会是:
        # tensor([[1.0, 0.5, 1.0], [1.0, 2.0, 1.0]])
    """
    if dtype is None: dtype = labels.dtype
    if device is None: device = labels.device
    weight_tensor = torch.ones_like(labels, dtype=dtype, device=device)
    if loss_weighting_map:
        for token_id, multiplier in loss_weighting_map.items():
            token_mask = (labels == token_id)
            weight_tensor[token_mask] = multiplier
    return weight_tensor


@torch.no_grad()
def build_ngram_unlikelihood_negative_samples(
    labels: torch.Tensor,
    repeat_ngram_n: int = 4,
    max_neg_next_tokens_per_pos: int = 12,
    ignore_token_ids: Optional[set[int]] = None,
) -> tuple[list[int], list[list[int]]]:
    """
    构造 UL 负采样位置及其对应的负例 token 列表（展平索引），标记出哪些位置有哪些负样本 token。
    该方法对于每个监督位置，查看其前面的 repeat_ngram_n-1 个 token 作为 prefix，统计该 prefix 在更早位置上出现过哪些 next-token，
    然后将这些 next-token（排除当前位置正确 token）作为负例。
    这样可以惩戒模型生成那些在上下文中已经出现过的重复内容，从而减少重复生成的可能性。
    缺陷是如果重复内容不存在于 labels 中（模型生成了重复内容），则无法惩戒到，对此问题应使用 recent-token UL （build_recent_token_negative_samples）。
    Args:
        labels (torch.Tensor): 真实标签张量，形状为 (batch_size, seq_length)。
        repeat_ngram_n (int): 用于负采样的 n-gram 大小。更长的 n-gram 会在更远距离上进行匹配。
        max_neg_next_tokens_per_pos (int): 每个位置允许的最大负样本 token 数量。也就是目标位置最大惩戒多少个可能造成重复的 token。
        ignore_token_ids (set[int], optional): 要忽略的 token ID 集合，这些 token 会从监督区间剔除（既不作为采样位置，也不参与 prefix 统计）。
    Returns:
        tuple[list[int], list[list[int]]]:
            - 负样本位置的**展平**索引列表。
            - 每个位置对应的负样本 token ID 列表。
    """
    if repeat_ngram_n < 2: raise ValueError('repeat_ngram_n must be at least 2.')

    labels_cpu = labels.detach().cpu()
    supervised = torch.ones_like(labels_cpu, dtype=torch.bool)  # [batch_size, seq_len]: True = 监督区间
    if ignore_token_ids is not None:
        for token_id in ignore_token_ids: supervised &= (labels_cpu != token_id)

    neg_pos_flat: list[int] = []
    neg_ids_list: list[list[int]] = []
    for batch_i in range(supervised.size(0)):  # 遍历每个 batch
        neg_indices = torch.nonzero(supervised[batch_i], as_tuple=False).squeeze(-1)  # [num_neg_tokens]，每个True的位置索引，这一步虽然可以在for之前做，但是无法在进入for循环后高效地筛选出每个batch_i对应的neg_indices，反而需要大量遍历，所以放在这里
        if neg_indices.numel() < 2: continue  # 负样本太少，跳过

        breaks = torch.nonzero(neg_indices[1:] != neg_indices[:-1] + 1, as_tuple=False).squeeze(-1) + 1
        starts = torch.cat([neg_indices[:1], neg_indices[breaks]])
        ends = torch.cat([neg_indices[breaks - 1], neg_indices[-1:]])

        row_tokens = labels_cpu[batch_i].tolist()  # 当前行的所有 token 列表
        segments = list(zip(starts.tolist(), ends.tolist()))  # [(seg_start, seg_end), ...]
        for seg_start_pos, seg_end_pos in segments:
            seg_length = seg_end_pos - seg_start_pos + 1
            if seg_length < 2: continue  # 片段太短，跳过
            n_eff = min(repeat_ngram_n, seg_length)

            # 段内 token 序列（全是监督 token，含 eot）
            seg_tokens = row_tokens[seg_start_pos: seg_end_pos + 1]
            prefix_to_next: dict[tuple[int, ...], set[int]] = {}  # 记录 repeat_ngram_n - 1 个上文 token 到 next-token 的映射，例如：{(这,是,一,个,) => {测(试),人(工),...}, ...}
            hist: deque[int] = deque(maxlen=n_eff - 1)  # 最近 n-1 个 token（来自 labels，本段内全是监督 token）
            for offset, current_token_id in enumerate(seg_tokens):  # 遍历当前段内的每个 token
                pos = seg_start_pos + offset  # 映射回原序列位置，pos = 起始位置 + 偏移量

                # 基于 hist 生成负例（此时 prefix_to_next 只包含“更早位置”的统计）
                # 如果记录的历史token数量足够（此处不可能出现大于情况，因为 hist 的 maxlen 就是 n-1，只是保险起见）
                if len(hist) >= (n_eff - 1):
                    prefix = tuple(hist)  # (token_id1, token_id2, ..., token_id_{n-1})
                    cand = prefix_to_next.get(prefix)  # 会被匹配的示例：A B C D  A B C E  A B C F -> prefix=(A,B,C) => cand={D,E,F}
                    if cand:
                        # 剔除当前位置的正确 token id，剩下的都是负例候选
                        cands = [x for x in cand if x != current_token_id]  # 该 prefix 历史出现过的 next-token 候选集合
                        if cands:
                            cands = cands[:max_neg_next_tokens_per_pos]
                            # 不够的用 -1 补齐
                            if len(cands) < max_neg_next_tokens_per_pos:
                                cands += [-1] * (max_neg_next_tokens_per_pos - len(cands))
                            neg_pos_flat.append(batch_i * labels_cpu.size(1) + pos)  # 记录位置（展平后的）
                            neg_ids_list.append(cands)  # 记录负例 token id 列表

                    prefix_to_next.setdefault(prefix, set()).add(current_token_id)

                # 更新 hist（滑窗推进）
                hist.append(current_token_id)

    return neg_pos_flat, neg_ids_list


@torch.no_grad()
def build_recent_unlikelihood_negative_samples(
    labels: torch.Tensor,
    recent_window_size: int = 64,
    max_neg_next_tokens_per_pos: int = 8,
    ignore_token_ids: Optional[set[int]] = None,
) -> tuple[list[int], list[list[int]]]:
    """
    构造 UL 负采样位置及其对应的负例 token 列表（展平索引），标记出哪些位置有哪些负样本 token。
    该方法对于每个监督位置，查看其前面的 recent_window_size 个 token，
    查看前 recent_window_size 个 token，把这些 token（去重，排除当前 gold）作为负例。
    这样可以惩戒模型生成那些在上下文中已经出现过的重复内容，从而减少重复生成的可能性。
    Args:
        labels (torch.Tensor): 真实标签张量，形状为 (batch_size, seq_length)。
        recent_window_size (int): 用于负采样的最近窗口大小。更大的窗口会覆盖更远距离的上下文。
        max_neg_next_tokens_per_pos (int): 每个位置允许的最大负样本 token 数量。也就是目标位置最大惩戒多少个可能造成重复的 token。
        ignore_token_ids (set[int], optional): 要忽略的 token ID 集合，这些 token 会从监督区间剔除（既不作为采样位置，也不参与 prefix 统计）。
    Returns:
        tuple[list[int], list[list[int]]]:
            - 负样本位置的**展平**索引列表。
            - 每个位置对应的负样本 token ID 列表。
    """
    labels_cpu = labels.detach().cpu()
    supervised = torch.ones_like(labels_cpu, dtype=torch.bool)
    if ignore_token_ids is not None:
        for tid in ignore_token_ids:
            supervised &= (labels_cpu != tid)

    neg_pos_flat: list[int] = []
    neg_ids_list: list[list[int]] = []
    for batch_i in range(supervised.size(0)):  # 遍历每个 batch
        neg_indices = torch.nonzero(supervised[batch_i], as_tuple=False).squeeze(-1)  # [num_neg_tokens]，每个True的位置索引
        if neg_indices.numel() < 2: continue  # 负样本太少，跳过

        # 连续段切分
        breaks = torch.nonzero(neg_indices[1:] != neg_indices[:-1] + 1, as_tuple=False).squeeze(-1) + 1
        starts = torch.cat([neg_indices[:1], neg_indices[breaks]])
        ends = torch.cat([neg_indices[breaks - 1], neg_indices[-1:]])

        row_tokens = labels_cpu[batch_i].tolist()  # 当前行的所有 token 列表
        segments = list(zip(starts.tolist(), ends.tolist()))  # [(seg_start, seg_end), ...]
        for seg_start_pos, seg_end_pos in segments:
            seg_length = seg_end_pos - seg_start_pos + 1
            if seg_length < 2: continue  # 片段太短，跳过

            # 段内 token 序列（全是监督 token，含 eot）
            seg_tokens = row_tokens[seg_start_pos: seg_end_pos + 1]

            recent: deque[int] = deque(maxlen=recent_window_size)

            for offset, current_token_id in enumerate(seg_tokens):  # 遍历当前段内的每个 token
                pos = seg_start_pos + offset  # 映射回原序列位置，pos = 起始位置 + 偏移量

                # 从 recent 中挑负例：按“最近优先”，去重，排除当前 gold
                if len(recent) > 0:
                    seen = set()
                    cands: list[int] = []
                    for token_id in reversed(recent):
                        if token_id != current_token_id and token_id not in seen:
                            seen.add(token_id)
                            cands.append(token_id)
                            if len(cands) >= max_neg_next_tokens_per_pos:
                                break
                    if cands:
                        # 不够的用 -1 补齐
                        if len(cands) < max_neg_next_tokens_per_pos:
                            cands += [-1] * (max_neg_next_tokens_per_pos - len(cands))
                        neg_pos_flat.append(batch_i * labels_cpu.size(1) + pos)  # 记录位置（展平后的）
                        neg_ids_list.append(cands)  # 记录负例 token id 列表

                # 更新 recent（滑窗推进）
                recent.append(current_token_id)

    return neg_pos_flat, neg_ids_list


def compute_unlikelihood_loss_from_negatives(
    predictions: torch.Tensor,
    neg_pos_flat: list[int],
    neg_ids_list: list[list[int]],
):
    """
    计算 UL 损失，基于预先构造好的负采样位置及其对应的负例 token 列表。
    Args:
        predictions (torch.Tensor): 预测 logits，形状为 (batch_size * seq_length, vocab_size)。
        neg_pos_flat (list[int]): 负样本位置的展平索引列表。
        neg_ids_list (list[list[int]]): 每个位置对应的负样本 token ID 列表。
    Returns:
        torch.Tensor: 计算得到的 UL 损失值（标量）。
    """
    # 检查 predictions 维度是否正确
    if predictions.dim() != 2:
        raise ValueError(f'Predictions tensor must be 2D (batch_size * seq_length, vocab_size), but got shape {predictions.shape}.')

    # 计算 UL loss（稀疏 gather）
    neg_pos = torch.tensor(neg_pos_flat, device=predictions.device, dtype=torch.long)  # [N], N = 负例总数，label展平后的位置索引
    neg_ids = torch.tensor(neg_ids_list, device=predictions.device, dtype=torch.long)  # [N, K], K = 每个位置的负例
    valid = (neg_ids >= 0)  # [N, K], bool 张量，标记有效的负例位置，因为使用了 -1 补齐

    log_z = torch.logsumexp(
        input=predictions.index_select(0, neg_pos),
        dim=-1,
        keepdim=True
    )  # [N, 1]
    neg_logits = predictions[neg_pos.unsqueeze(1), neg_ids.clamp_min(0)]  # [N, K]
    p_neg = torch.exp(neg_logits - log_z).clamp_max(1.0 - 1e-6)  # [N, K]
    p_neg = p_neg * valid.float()  # 将无效位置的概率置零

    ul_matrix = -torch.log1p(-p_neg)  # [N, K]
    return ul_matrix.sum() / valid.float().sum().clamp_min(1.0)
