import numpy
import torch


def build_span_mask(
    input_ids: numpy.ndarray | list[int] | list[list[int]] | torch.Tensor,
    start_token_id: int,
    end_token_id: int,
    include_start_token: bool = True,
    include_end_token: bool = True,
) -> numpy.ndarray:
    """
    构建 span mask：由 start_token_id 与其后最近的 end_token_id 定义一个区间为有效区域，其他区域为 ignore。
    Args:
        input_ids (numpy.ndarray | list[int] | list[list[int]] | torch.Tensor): 输入的 token id 序列，shape (seq_len,) 或 (batch_size, seq_len)
        start_token_id (int): 起始 token id
        end_token_id (int): 结束 token id
        include_start_token (bool): 是否包含起始 token 在内
        include_end_token (bool): 是否包含结束 token 在内
    Returns:
        span_mask (numpy.ndarray): 布尔型掩码，shape (seq_len,) 或 (batch_size, seq_len)。True 表示有效区域，False 表示 ignore 区域。
    例子:
        input_ids = [10, 20, 30, 40, 50, 60, 70]
        start_token_id = 20
        end_token_id = 60
        输出: [False, True, True, True, True, False, False] (假设 include_start_token=True, include_end_token=False)
    解释: 从第一个 20 开始，到第一个 60 结束，形成有效区域。
    备注: 假设输入中 start_token_id 和 end_token_id 成对出现，不会嵌套。
    """
    if isinstance(input_ids, list):
        input_ids = numpy.array(input_ids)
    elif isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.detach().cpu().numpy()
    # 检查输入形状
    if input_ids.ndim > 2: raise ValueError("Input ids should be 1D or 2D array.")
    if input_ids.ndim == 2:
        # 从第一个维度切片调用自身再拼接
        return numpy.stack([build_span_mask(
            input_ids[i],
            start_token_id,
            end_token_id,
            include_start_token,
            include_end_token,
        ) for i in range(input_ids.shape[0])], axis=0)

    # True = 有效, False = ignore(无效，pad)
    span_mask = numpy.zeros_like(input_ids, dtype=bool)

    start_pos = numpy.where(input_ids == start_token_id)[0]
    end_pos = numpy.where(input_ids == end_token_id)[0]

    for pos in start_pos:
        start = pos
        if not include_start_token: start += 1
        # 找到对应的 eot 标记位置
        end_candidates = numpy.searchsorted(end_pos, start, side='left')
        if end_candidates >= len(end_pos):
            raise ValueError(f"No matching end token found for start token at position {pos}")
        end = end_pos[end_candidates]
        if include_end_token: end += 1

        span_mask[start:end] = True

    return span_mask


def apply_attention_mask_to_hidden_states(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    keep_shape: bool = False,
) -> torch.Tensor:
    """
    将注意力掩码应用于隐藏状态。
    Args:
        hidden_states (torch.Tensor): 隐藏状态，shape (batch_size, seq_len, hidden_dim)
        attention_mask (torch.Tensor):
            注意力掩码，shape (batch_size, seq_len)。
            BoolTensor 时 True 表示有效位置，False 表示无效位置。
            FloatTensor 时 1.0 表示有效位置，0.0 表示无效位置。
        keep_shape (bool): 是否保持原始形状。如果 False 则返回一个压缩后的张量，仅包含有效位置的隐藏状态。
    Returns:
        masked_hidden_states (torch.Tensor):
            应用掩码后的隐藏状态：
            - 如果 keep_shape=True，形状与输入相同，掩码位置的隐藏状态为 0。
            - 如果 keep_shape=False，形状为 (num_valid_positions, hidden_dim)，仅包含有效位置的隐藏状态。
    """
    if hidden_states.ndim != 3: raise ValueError("Hidden states should be a 3D tensor.")
    if attention_mask.ndim != 2: raise ValueError("Attention mask should be a 2D tensor.")
    if hidden_states.shape[0] != attention_mask.shape[0] or hidden_states.shape[1] != attention_mask.shape[1]:
        raise ValueError("Batch size and sequence length of hidden states and attention mask must match.")

    mask = attention_mask
    if mask.dtype != torch.bool: mask = mask.bool()
    if keep_shape: masked_hidden_states = hidden_states.masked_fill(~mask.unsqueeze(-1), 0)  # 将掩码位置的隐藏状态设置为 0
    else: masked_hidden_states = hidden_states[mask]

    return masked_hidden_states
