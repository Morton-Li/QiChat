from typing import Literal

from transformers import PretrainedConfig


class QiChatConfig(PretrainedConfig):
    model_type = "QiChat"

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 1024,
        num_hidden_layers: int = 24,
        n_heads: int = 16,
        d_kv: int = 64,  # d_model // n_heads
        n_kv_heads: int = 4,
        d_ff: int = 2560,
        emb_dropout: float = 0.01,
        tie_word_embeddings: bool = True,
        rope_theta: int = 10000,
        rms_norm_eps: float = 1e-6,
        attn_dropout_rate: float = 0.01,
        ffn_activation: Literal['silu', 'swish', 'gelu', 'relu'] = 'silu',  # SwiGLU/GEGLU/ReGLU
        attn_implementation: Literal['eager', 'sdpa', 'flash_attention_2', 'flash_attention_3'] = 'sdpa',
        initializer_range: float = 0.01,
        return_dict: bool = True,
        tokenizer_class: str = 'QiTianTokenizerFast',
        bos_token_id: int = 0,
        eos_token_id: int = 1,
        pad_token_id: int = 3,
        dtype: str = 'bfloat16',
        **kwargs,
    ):
        """
        QiChat model configuration
        Args:
            vocab_size (int): Vocabulary size of the model
            d_model (int): Dimension of the model
            num_hidden_layers (int): Number of layers in the model
            n_heads (int): Number of attention heads
            d_kv (int): Dimension of key/value vectors
            n_kv_heads (int): Number of key/value heads
            d_ff (int): Dimension of feed-forward layer
            emb_dropout (float): Dropout probability for embeddings
            max_position_embeddings (int): Maximum number of position embeddings
            tie_word_embeddings (bool): Whether to tie input and output embeddings
            rope_theta (int): Base frequency for rotary position embeddings
            rms_norm_eps (float): Epsilon for RMS normalization
            attn_dropout_rate (float): Dropout probability for attention layers
            ffn_activation (str): Activation function for feed-forward layers
            attn_implementation (str): Attention implementation to use
            initializer_range (float): Standard deviation for weight initialization
            return_dict (bool): Whether to return outputs as a dict
            tokenizer_class (str): Tokenizer class name
            bos_token_id (int): Beginning of sequence token ID
            eos_token_id (int): End of sequence token ID
            pad_token_id (int): Padding token ID
            decoder_start_token_id (int): Decoder start token ID
            dtype (str): Data type for model weights
        """
        super().__init__(
            tokenizer_class=tokenizer_class,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            dtype=dtype,  # 会自动转 torch.dtype
            is_decoder=True,
            attn_implementation=attn_implementation,
            return_dict=return_dict,
            **kwargs,
        )

        self.vocab_size = vocab_size
        assert d_model % 2 == 0, "d_model must be an even number"
        self.d_model = d_model
        self.num_hidden_layers = num_hidden_layers
        assert d_model % n_heads == 0, "d_model must be divisible by num_heads"
        self.n_heads = n_heads
        assert d_kv == d_model // n_heads, "d_kv must equal d_model // n_heads"
        self.d_kv = d_kv
        assert n_kv_heads <= n_heads, "n_kv_heads must be less than or equal to num_heads"
        assert n_kv_heads > 0, "n_kv_heads must be greater than 0"
        assert n_heads % n_kv_heads == 0, "num_heads must be divisible by n_kv_heads"
        self.n_kv_heads = n_kv_heads
        self.d_ff = d_ff

        self.emb_dropout = emb_dropout
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta

        self.rms_norm_eps = rms_norm_eps
        self.attn_dropout_rate = attn_dropout_rate
        self.ffn_activation = ffn_activation

        self.initializer_range = initializer_range
