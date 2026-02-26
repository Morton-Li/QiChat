from typing import Optional

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import PreTrainedModel, PretrainedConfig, Cache, DynamicCache, GenerationMixin
from transformers.activations import ACT2FN
from transformers.integrations.sdpa_attention import repeat_kv
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.utils.generic import maybe_autocast

from .configuration import QiChatConfig


class QiChatWordEmbeddings(nn.Module):
    """ QiChat Word Embeddings module"""
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        pad_token_id: Optional[int] = None,
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=pad_token_id
        )
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, input_ids: torch.LongTensor, mask: Optional[torch.BoolTensor] = None) -> torch.FloatTensor:
        """
        Forward pass for the masked normalized embedding layer
        Args:
            input_ids (torch.LongTensor): Input token IDs of shape (batch_size, seq_len)
            mask (Optional[torch.BoolTensor]): Attention mask of shape (batch_size, seq_len), where True indicates valid tokens
        Returns:
            torch.FloatTensor: Output embeddings of shape (batch_size, seq_len, d_model)
        """
        embeddings = self.word_embeddings(input_ids)  # [batch_size, seq_len, d_model]
        if mask is not None:
            if mask.shape[-1] != input_ids.shape[-1]:  # 自回归推理阶段
                mask = mask[:, -input_ids.shape[-1]:]  # 仅保留与当前输入对应的部分
            # Note: torch.all(mask) 会触发 GPU 同步所以不再进行该判断而是直接进行乘法操作，mask 全1 的情况下也不会改变结果
            # if not torch.all(mask):  # 仅在 mask 存在且非全1时才操作
            mask = mask.unsqueeze(-1).to(embeddings.dtype)  # [batch_size, seq_len, 1]
            embeddings = embeddings * mask  # Apply mask to embeddings
        return self.dropout(embeddings)


class QiChatRotaryPositionEmbedding(nn.Module):
    """ QiChat Rotary Position Embedding """
    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        dtype: torch.dtype = torch.bfloat16
    ):
        """
        Args:
            dim (int): Dimension of the model (must be even)
            base (float): Base frequency
            dtype (torch.dtype): Data type for the embeddings
        """
        super().__init__()
        assert dim % 2 == 0, "Rotary embedding requires even dimensions"
        self.output_dtype = dtype

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))  # [dim/2]
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.LongTensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            position_ids (torch.LongTensor):
                Position IDs of shape (batch_size, seq_len)
                prefill 时 seq_len 为完整序列长度，auto-regressive decode 时 seq_len 为当前输入序列长度（通常为1）
        Returns:
            cos (torch.float32): [batch_size, seq_len, dim/2]
            sin (torch.float32): [batch_size, seq_len, dim/2]
        """
        if position_ids.dim() != 2: raise ValueError(f"position_ids must be of shape (batch_size, seq_len), but got {position_ids.shape}")

        with maybe_autocast(device_type=position_ids.device.type, enabled=False):  # Force float32
            inv_freq_fp32 = self.inv_freq.to(device=position_ids.device, dtype=torch.float32)  # [dim/2]
            position_ids_fp32 = position_ids.to(device=position_ids.device, dtype=torch.float32)  # [batch_size, seq_len]

            freqs = position_ids_fp32[..., None] * inv_freq_fp32[None, None, :]  # [batch_size, seq_len, dim/2]
            cos, sin = freqs.cos(), freqs.sin()  # [batch_size, seq_len, dim/2]

        return cos.to(dtype=self.output_dtype), sin.to(dtype=self.output_dtype)


class QiChatPreTrainedModel(PreTrainedModel):
    """ QiChat Base Model """
    config_class = QiChatConfig

    _supports_flash_attn = True
    _supports_sdpa = True

    supports_gradient_checkpointing = True


def apply_rotary_pos_emb(x: torch.FloatTensor, cos: torch.FloatTensor, sin: torch.FloatTensor) -> torch.Tensor:
    """
    Apply RoPE (Rotary Position Embedding) to the input tensor.

    Args:
        x (`torch.FloatTensor`):
            The input tensor to which the rotary embedding will be applied.
            [batch_size, n_heads, seq_len, d_kv]
        cos (`torch.FloatTensor`):
            The cosine part of the rotary embedding.
            [batch_size, seq_len, dim/2]
        sin (`torch.FloatTensor`):
            The sine part of the rotary embedding.
            [batch_size, seq_len, dim/2]
    Returns:
        `torch.FloatTensor`: The tensor after applying the rotary position embedding.
    """
    batch_size, n_heads, seq_len, dim = x.shape
    if dim % 2 != 0:
        raise ValueError("The head dimension (d_kv) must be even for RoPE.")
    if cos.shape != (batch_size, seq_len, dim // 2) or sin.shape != (batch_size, seq_len, dim // 2):
        raise ValueError(f'cos and sin must have shape (batch_size, seq_len, dim/2), but got {cos.shape} and {sin.shape}')
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)  # [batch_size, 1, seq_len, dim/2]

    return torch.cat([
        x[..., :dim // 2] * cos - x[..., dim // 2:] * sin,
        x[..., dim // 2:] * cos + x[..., :dim // 2] * sin
    ], dim=-1)


class QiChatAttention(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int,
    ):
        """
        Args:
            config (PretrainedConfig): Model configuration object.
            layer_idx (int): Index of the current layer in the model.
        """
        super().__init__()

        self.config: PretrainedConfig = config
        self.layer_idx: int = layer_idx

        # attention_interface指定变量名
        self.num_key_value_groups: int = self.config.n_heads // self.config.n_kv_heads  # 每个 K/V 头对应的 Q 头数
        self.is_causal: bool = True

        # GQA（Grouped-Query Attention）
        proj_dim_qo = self.config.n_heads * self.config.d_kv  # Q 的输出维、O 的输入维
        proj_dim_kv = self.config.n_kv_heads * self.config.d_kv  # K、V 的输出维

        self.scaling = self.config.d_kv ** -0.5

        self.q_proj = nn.Linear(in_features=self.config.d_model, out_features=proj_dim_qo, bias=False)
        self.k_proj = nn.Linear(in_features=self.config.d_model, out_features=proj_dim_kv, bias=False)
        self.v_proj = nn.Linear(in_features=self.config.d_model, out_features=proj_dim_kv, bias=False)
        self.o_proj = nn.Linear(in_features=proj_dim_qo, out_features=self.config.d_model, bias=False)

        # QK-Norm
        self.q_norm = nn.RMSNorm(normalized_shape=self.config.d_kv, eps=self.config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(normalized_shape=self.config.d_kv, eps=self.config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        mask: Optional[torch.Tensor],
        cos_buf: torch.FloatTensor, sin_buf: torch.FloatTensor,
        use_cache: bool = False,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
    ) -> tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        """
        Forward pass for the QiChat attention layer.
        Args:
            hidden_states: [batch_size, seq_len, d_model]
            mask:
                attn_implementation in ['eager', 'sdpa'] and mask not None:
                    [batch_size, 1, seq_len, seq_len] (float tensor with 0.0 for valid positions and -inf for masked positions)
                attn_implementation in ['flash_attention_2', 'flash_attention_3'] and mask not None:
                    [batch_size, seq_len] (bool tensor with True for valid positions and False for masked positions)
                or None
            cos_buf: [batch_size, seq_len, dim/2]
            sin_buf: [batch_size, seq_len, dim/2]
            use_cache (bool): Whether to use caching for past key/values
            position_ids (Optional[torch.LongTensor]): Position IDs for RoPE
            past_key_values: Cache object for storing past key/values
            cache_position: [seq_len] (long tensor) indicating the current position for caching
            output_attentions (bool): Whether to output attention weights
        Returns:
            attn_output: [batch_size, seq_len, d_model]
            attn_weights: [batch_size, n_heads, seq_len, seq_len] or None
        """
        if self.training:
            if use_cache: raise RuntimeError("use_cache=True is not supported during training.")

        batch_size, seq_len, _ = hidden_states.size()

        if len({cos_buf.dtype, sin_buf.dtype, hidden_states.dtype}) > 1:
            raise ValueError(f'cos_buf, sin_buf and hidden_states must have the same dtype, but got {cos_buf.dtype}, {sin_buf.dtype} and {hidden_states.dtype}')
        if cos_buf.shape != sin_buf.shape:
            raise ValueError(f'cos_buf and sin_buf must have the same shape, but got {cos_buf.shape} and {sin_buf.shape}')
        if cos_buf.shape != (batch_size, seq_len, self.config.d_kv // 2):
            raise ValueError(f'cos_buf and sin_buf must have shape (batch_size, seq_len, d_kv/2), but got {cos_buf.shape} and {sin_buf.shape}')

        # QKV projection
        query_states = self.q_norm(self.q_proj(hidden_states).view(batch_size, seq_len, self.config.n_heads, self.config.d_kv)).transpose(1, 2)       # [batch_size, n_heads, seq_len, d_kv]
        key_states = self.k_norm(self.k_proj(hidden_states).view(batch_size, seq_len, self.config.n_kv_heads, self.config.d_kv)).transpose(1, 2)    # [batch_size, n_kv_heads, seq_len, d_kv]
        value_states = self.v_proj(hidden_states).view(batch_size, seq_len, self.config.n_kv_heads, self.config.d_kv).transpose(1, 2)    # [batch_size, n_kv_heads, seq_len, d_kv]

        # Apply RoPE to Q and K
        query_states = apply_rotary_pos_emb(x=query_states, cos=cos_buf, sin=sin_buf)
        key_states = apply_rotary_pos_emb(x=key_states, cos=cos_buf, sin=sin_buf)

        # Handle caching for auto-regressive generation
        if use_cache:
            if not isinstance(past_key_values, Cache): raise ValueError("past_key_values must be a Cache object")
            if cache_position.dtype != torch.long: raise ValueError("cache_position must be a LongTensor")

            # Update K and V with cached values
            key_states, value_states = past_key_values.update(
                key_states=key_states, value_states=value_states,
                layer_idx=self.layer_idx,
                cache_kwargs={'cache_position': cache_position, 'position_ids': position_ids},
            )  # [batch_size, n_kv_heads, seq_len, d_kv]

        if self.config._attn_implementation == 'eager':
            if self.num_key_value_groups > 1:
                key_states = repeat_kv(key_states, self.num_key_value_groups)  # [batch_size, n_heads, seq_len, d_kv]
                value_states = repeat_kv(value_states, self.num_key_value_groups)  # [batch_size, n_heads, seq_len, d_kv]

            # Compute attention scores
            attn_scores = torch.matmul(query_states, key_states.transpose(2, 3))  # [batch_size, n_heads, seq_len, seq_len]
            attn_scores = attn_scores * self.scaling  # [batch_size, n_heads, seq_len, seq_len]
            if mask is None:
                # 构造上三角掩码
                min_val = torch.finfo(attn_scores.dtype).min
                mask = torch.triu(
                    torch.full((seq_len, seq_len), min_val, device=attn_scores.device, dtype=attn_scores.dtype),
                    diagonal=1
                )[None, None, :, :]  # [1, 1, seq_len, seq_len]  上三角=min_val，下三角=0（默认就是0）
            elif not mask.is_floating_point(): raise ValueError(f'mask must be a FloatTensor, but got {mask.dtype}')

            attn_scores = attn_scores + mask  # [batch_size, n_heads, seq_len, seq_len]

            attn_weights = nn.functional.softmax(input=attn_scores, dim=-1)  # [batch_size, n_heads, seq_len, seq_len]
            attn_weights = nn.functional.dropout(input=attn_weights, p=self.config.attn_dropout_rate, training=self.training)  # [batch_size, n_heads, seq_len, seq_len]

            attn_output = torch.matmul(attn_weights, value_states)  # [batch_size, n_heads, seq_len, d_kv]
            attn_output = attn_output.transpose(1, 2).contiguous()  # [batch_size, seq_len, n_heads, d_kv]
        else:
            # SDPA, Flash Attention, Flex Attention 会自动处理 repeat_kv
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
            attn_output, attn_weights = attention_interface(
                module=self,
                query=query_states, key=key_states, value=value_states,
                attention_mask=mask,  # [batch_size, 1, seq_len, seq_len] or [batch_size, seq_len] or None
                dropout=self.config.attn_dropout_rate if self.training else 0.0,
                scaling=self.scaling,
            )  # [batch_size, seq_len, n_heads, d_kv], None

        if attn_output.shape != (batch_size, seq_len, self.config.n_heads, self.config.d_kv):
            raise ValueError(f"Expected attn_output shape {(batch_size, seq_len, self.config.n_heads, self.config.d_kv)}, but got {attn_output.shape}")

        attn_output = attn_output.view(batch_size, attn_output.size(1), self.config.n_heads * self.config.d_kv)  # [batch_size, seq_len, n_heads * d_kv]

        attn_output = self.o_proj(attn_output)  # [batch_size, seq_len, d_model]

        return attn_output, attn_weights if output_attentions else None


class GLUFeedForward(nn.Module):
    """ 带门控线性单元 (GLU) 的前馈神经网络模块 """
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        activation: str
    ):
        """
        Args:
            d_model (int): 输入和输出的隐藏状态维度。
            d_ff (int): 前馈网络的中间层维度。
            activation (str): 激活函数的名称（如 "relu", "gelu" 等）。
        """
        super().__init__()
        if activation not in ACT2FN.keys():
            raise ValueError(f"activation {activation} is not supported.")

        self.gate_proj = nn.Linear(in_features=d_model, out_features=d_ff, bias=False)
        self.up_proj = nn.Linear(in_features=d_model, out_features=d_ff, bias=False)
        self.down_proj = nn.Linear(in_features=d_ff, out_features=d_model, bias=False)
        self.activation = ACT2FN[activation]

    def forward(self, hidden_states):
        """
        前馈网络的前向传播，使用 GLU 机制。
        Args:
            hidden_states (torch.FloatTensor): 输入的隐藏状态，形状为 (batch_size, seq_length, d_model)。
        Returns:
            torch.FloatTensor: 输出的隐藏状态，形状为 (batch_size, seq_length, d_model)。
        """
        gate_hidden_states = self.gate_proj(hidden_states)
        hidden_states = self.up_proj(hidden_states)
        hidden_states = self.activation(gate_hidden_states) * hidden_states
        hidden_states = self.down_proj(hidden_states)

        return hidden_states


class QiChatBlock(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int,
    ):
        """
        Args:
            config (PretrainedConfig): Model configuration object.
            layer_idx (int): Index of the current layer in the model.
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Note:
        # nn.RMSNorm 在 kernel 层面已进行单精度计算以保证数值稳定性
        # 详见：
        # CPU: pytorch/aten/src/ATen/native/layer_norm.cpp
        # CUDA: pytorch/aten/src/ATen/native/cuda/layer_norm_kernel.cu
        # 精度定义在 pytorch/aten/src/ATen/AccumulateType.h
        self.input_layer_norm = nn.RMSNorm(normalized_shape=self.config.d_model, eps=self.config.rms_norm_eps)
        self.self_attention = QiChatAttention(
            config=self.config,
            layer_idx=layer_idx,
        )
        self.output_layer_norm = nn.RMSNorm(normalized_shape=self.config.d_model, eps=self.config.rms_norm_eps)
        self.ffn = GLUFeedForward(d_model=self.config.d_model, d_ff=self.config.d_ff, activation=self.config.ffn_activation)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        mask: torch.Tensor,
        cos_buf: torch.FloatTensor, sin_buf: torch.FloatTensor,
        use_cache: bool = False,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
    ) -> tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        """
        Args:
            hidden_states (torch.FloatTensor): Input hidden states of shape (batch_size, seq_length, d_model)
            mask (torch.Tensor): Attention mask of shape (batch_size, 1, seq_len, seq_len)
            cos_buf (torch.FloatTensor): Cosine positional embeddings buffer of shape [batch_size, seq_len, dim/2]
            sin_buf (torch.FloatTensor): Sine positional embeddings buffer of shape [batch_size, seq_len, dim/2]
            use_cache (bool): Whether to use caching for past key/values
            position_ids (Optional[torch.LongTensor]): Position IDs for RoPE
            past_key_values (Optional[Cache]): Past key/values for caching
            cache_position (Optional[torch.LongTensor]): Cache positions for dynamic caching
            output_attentions (bool): Whether to output attention weights
        Returns:
            torch.FloatTensor: Output hidden states of shape (batch_size, seq_length, d_model)
            Optional[torch.FloatTensor]: Attention weights if output_attentions is True
        """
        if self.training:
            if use_cache: raise RuntimeError("use_cache=True is not supported during training.")
        if not use_cache:
            past_key_values, cache_position, position_ids = None, None, None

        # Self-Attention
        normed_states = self.input_layer_norm(hidden_states)
        attention_output, attention_weights = self.self_attention(
            hidden_states=normed_states,
            mask=mask,
            cos_buf=cos_buf, sin_buf=sin_buf,
            use_cache=use_cache,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + attention_output  # Residual connection

        # Feed Forward Network
        normed_states = self.output_layer_norm(hidden_states)
        ffn_output = self.ffn(normed_states)
        hidden_states = hidden_states + ffn_output  # Residual connection

        return hidden_states, attention_weights


class QiChatModel(QiChatPreTrainedModel):
    """ QiChat Model """

    _no_split_modules = ['QiChatBlock']

    def __init__(self, config: PretrainedConfig):
        """
        Args:
            config (PretrainedConfig): Model configuration object.
        """
        super().__init__(config=config)

        self.word_emb = QiChatWordEmbeddings(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            pad_token_id=config.pad_token_id,
            dropout_prob=config.emb_dropout,
        )

        self.rotary_pos_emb = QiChatRotaryPositionEmbedding(
            dim=config.d_kv,
            base=config.rope_theta,
            dtype=config.dtype,  # 输出精度，该模块内部会提升精度以保证数值稳定性，然后再转换回与模型一致的精度参与计算
        )

        self.blocks = nn.ModuleList([
            QiChatBlock(config=config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])

        self.final_layer_norm = nn.RMSNorm(normalized_shape=config.d_model, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module: return self.word_emb.word_embeddings
    def set_input_embeddings(self, value: nn.Module): self.word_emb.word_embeddings = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        use_cache: bool = False,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_hidden_states: bool = True,
        output_attentions: bool = False,
    ) -> BaseModelOutputWithPast:
        """
        Forward pass of the QiChat model.
        Args:
            input_ids (Optional[torch.LongTensor]): Input token IDs of shape (batch_size, seq_length)
            inputs_embeds (Optional[torch.FloatTensor]): Input token embeddings of shape (batch_size, seq_length, d_model)
            attention_mask (Optional[torch.BoolTensor]): Attention mask of shape, where True indicates valid tokens (batch_size, seq_length)
            use_cache (bool): Whether to use caching for past key/values
            position_ids (Optional[torch.LongTensor]): Position IDs for RoPE, if None, will be generated based on attention_mask
            cache_position (Optional[torch.LongTensor]): Cache positions for dynamic caching
            past_key_values (Optional[Cache]): Past key/values for caching
            output_hidden_states (bool): Whether to output hidden states
            output_attentions (bool): Whether to output attentions
        Returns:
            BaseModelOutputWithPast: Model output containing last hidden state and past key/values
        """
        if self.training:
            if use_cache: raise RuntimeError("use_cache=True cannot be used during training")

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is not None:
            batch_size, seq_len = input_ids.shape
            use_device = input_ids.device
        elif inputs_embeds is not None:
            batch_size, seq_len, emb_dim = inputs_embeds.shape
            if emb_dim != self.config.d_model:
                raise ValueError(f"The dimension of the inputs_embeds ({emb_dim}) does not match the model's d_model ({self.config.d_model}).")
            use_device = inputs_embeds.device
        else: raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None: attention_mask = torch.ones((batch_size, seq_len), device=use_device, dtype=torch.bool)
        elif attention_mask.dtype != torch.bool: attention_mask = attention_mask.to(dtype=torch.bool)

        if input_ids is not None:
            hidden_states = self.word_emb(
                input_ids=input_ids,
                mask=attention_mask,  # 该词嵌入模块可选接受mask以跳过注意力看不到的位置的嵌入计算，True为有效位置
            )  # [batch_size, seq_len, d_model]
        elif inputs_embeds is not None: hidden_states = inputs_embeds
        else: raise ValueError("You have to specify either input_ids or inputs_embeds")

        if position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1  # [batch_size, seq_len] 位置索引从0开始，padding部分为-1, 例如：[-1, -1, -1, 0, 1, 2, 3, 4]
            position_ids.masked_fill_(attention_mask == 0, 0)  # [batch_size, seq_len] 将padding部分位置索引设为0: [0, 0, 0, 0, 1, 2, 3, 4]
            # Note: HF 在 prepare_inputs_for_generation 中采用 1 进行填充，本质上不影响计算结果
            if position_ids.size(-1) != seq_len: position_ids = position_ids[:, -seq_len:]  # 仅保留与当前输入对应的部分

        cos_buf, sin_buf = self.rotary_pos_emb(position_ids=position_ids)  # [batch_size, seq_len, dim/2]

        if use_cache:
            if past_key_values is None: past_key_values = DynamicCache(config=self.config)
            elif not isinstance(past_key_values, Cache): raise ValueError(f'past_key_values must be a DynamicCache object, but got {type(past_key_values)}')

            if cache_position is None:
                past_key_values_length = past_key_values.get_seq_length()  # 已缓存的序列长度
                # 当前输入序列（prefill 或 decode 新追加的 tokens）在整个上下文中的 绝对位置索引（starting from 0）
                # 意味着本次需要缓存的位置
                cache_position = torch.arange(
                    past_key_values_length, past_key_values_length + seq_len, device=use_device
                )  # [seq_len]

            attention_mask = create_causal_mask(
                config=self.config,
                input_embeds=hidden_states,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
        else:
            past_key_values, cache_position = None, None

            attention_mask = create_causal_mask(
                config=self.config,
                input_embeds=hidden_states,
                attention_mask=attention_mask,
                cache_position=torch.arange(0, seq_len, device=use_device),  # [seq_len] 绝对位置索引,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
        # attn_implementation in [eager, flex_attention] 时 attention_mask 是 FloatTensor 类型，被遮挡位置为 -inf（根据dtype自动调整），有效位置为 0.0
        # attn_implementation in [sdpa, flash_attention_2, flash_attention_3] 时 attention_mask 是 BoolTensor 类型，True 表示有效位置，False 表示被遮挡位置
        # 且 attn_implementation in [flash_attention_2, flash_attention_3] 时，输入 attention_mask 经过 create_causal_mask 后 = 输出 attention_mask

        if len({cos_buf.dtype, sin_buf.dtype, hidden_states.dtype}) > 1:
            # 只会出现在提供 inputs_embeds 时
            raise ValueError(f"Data type of inputs_embeds ({hidden_states.dtype}) does not match model dtype ({self.config.dtype})")

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i, block in enumerate(self.blocks):
            if output_hidden_states: all_hidden_states = all_hidden_states + (hidden_states,)

            if self.training and self.gradient_checkpointing:
                hidden_states, attention_weights = checkpoint(
                    block,
                    hidden_states,
                    attention_mask,
                    cos_buf, sin_buf,
                    False,
                    None,  # position_ids 只被 past_key_values 消费，当 past_key_values 为 None 时没必要传递 position_ids
                    None,  # past_key_values cannot be used during training
                    None,  # cache_position 只被 past_key_values 消费，当 past_key_values 为 None 时没必要传递 cache_position
                    output_attentions,
                    use_reentrant=False,  # 显式禁用重入式引擎
                )
            else:
                hidden_states, attention_weights = block(
                    hidden_states=hidden_states,
                    mask=attention_mask,
                    cos_buf=cos_buf, sin_buf=sin_buf,
                    use_cache=use_cache,
                    position_ids=position_ids,
                    past_key_values=past_key_values,  # 每一步会原地更新past_key_values
                    cache_position=cache_position,
                    output_attentions=output_attentions,
                )

            if output_attentions: all_attentions = all_attentions + (attention_weights,)

        hidden_states = self.final_layer_norm(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attentions
        )

    @property
    def dummy_inputs(self) -> dict[str, torch.Tensor]:
        """ Dummy inputs for testing and tracing. """
        return {
            "input_ids": torch.tensor([
                [   0,   47, 8260,   19, 2071,    8,    1,    2,    2,    2,    2,    2],
                [   0, 6910, 2760,  320, 2607, 7868,    1,    2,    2,    2,    2,    2],
                [   0, 2511, 1765, 5067, 8440, 6497, 3055, 1661,  254, 9995,  364,    1]
            ], dtype=torch.long, device=self.device),
            "attention_mask": torch.tensor([
                [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ], dtype=torch.bool, device=self.device),
        }


class QiChatCausalLMHead(nn.Module):
    """ Head for causal language modeling (next-token prediction) """
    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        bias: bool = False,
    ) -> None:
        """
        Args:
            d_model: Dimension of the model hidden states.
            vocab_size: Size of the vocabulary.
            bias: Whether to include a bias term in the projection layer.
        """
        super().__init__()
        self.dense = nn.Linear(in_features=d_model, out_features=vocab_size, bias=bias)

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            hidden_states: Tensor of shape (batch_size, seq_length, d_model)
        Returns:
            logits: Tensor of shape (batch_size, seq_length, vocab_size)
        """
        assert hidden_states.shape[-1] == self.dense.in_features, f"Expected hidden_states last dimension to be {self.dense.in_features}, but got {hidden_states.shape[-1]}"
        logits = self.dense(hidden_states)
        return logits


class QiChatForCausalLM(QiChatPreTrainedModel, GenerationMixin):
    """ QiChat model with a language modeling head for causal language modeling. """
    _tied_weights_keys = {'lm_head.dense.weight': 'model.word_emb.word_embeddings.weight'}

    def __init__(self, config: PretrainedConfig):
        super().__init__(config=config)

        self.model = QiChatModel(config=config)
        self.lm_head = QiChatCausalLMHead(
            d_model=config.d_model,
            vocab_size=config.vocab_size,
            bias=False
        )

        self.loss_type = "ForCausalLM"

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module: return self.model.get_input_embeddings()
    def set_input_embeddings(self, value: nn.Module): self.model.set_input_embeddings(value)
    def get_output_embeddings(self) -> nn.Module: return self.lm_head.dense
    def set_output_embeddings(self, value: nn.Module): self.lm_head.dense = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        use_cache: bool = False,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass of the QiChat model for causal language modeling.
        Args:
            input_ids (Optional[torch.LongTensor]): Input token IDs of shape (batch_size, seq_length)
            inputs_embeds (Optional[torch.FloatTensor]): Input token embeddings of shape (batch_size, seq_length, d_model)
            attention_mask (Optional[torch.BoolTensor]): Attention mask of shape (batch_size, seq_length)
            use_cache (bool): Whether to use caching for past key/values
            position_ids (Optional[torch.LongTensor]): Position IDs for RoPE (if not provided, will be generated based on attention_mask)
            cache_position (Optional[torch.LongTensor]): Cache positions for dynamic caching
            past_key_values (Optional[Cache]): Past key/values for caching
            labels (Optional[torch.LongTensor]): Labels for computing the language modeling loss
            output_hidden_states (bool): Whether to output hidden states
            output_attentions (bool): Whether to output attentions
            return_dict (bool): Whether to return a dict-like object
            **kwargs:
                - shift_labels: (Optional[torch.LongTensor]) Shifted labels for next-token prediction
                - num_items_in_batch: (Optional[torch.Tensor]) Number of items in the batch for loss calculation
                - loss_ignore_index: (Optional[int]) Index to ignore in the loss calculation
        Returns:
            CausalLMOutputWithPast: Model output containing logits and past key/values
        """
        if use_cache is True and self.training is True:
            raise RuntimeError("use_cache=True cannot be used during training")
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        if not return_dict: raise NotImplementedError("return_dict=False is not supported yet in QiChatForCausalLM")

        outputs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=use_cache,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=past_key_values,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        logits = self.lm_head(hidden_states=outputs.last_hidden_state)

        loss = None
        shift_labels = kwargs.pop('shift_labels', None)
        if labels is not None or shift_labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                shift_labels=shift_labels,
                vocab_size=logits.size(-1),  # 从 logits 获取而不是 config 中确保 vocab_size 真实可靠
                num_items_in_batch=kwargs.pop("num_items_in_batch", None),  # CE会自动计算，除非有特殊需求再传递 num_items_in_batch
                ignore_index=kwargs.pop('loss_ignore_index', -100)
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
