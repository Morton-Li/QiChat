import math

import torch


@torch.no_grad()
def collect_layer_grad_norms(model: torch.nn.Module) -> dict[str, float]:
    """
    Collect L2 norms of gradients for each layer in the model.
    Returns a dict mapping layer keys (e.g., "block_0", "emb", "final_ln") to their corresponding gradient norms.
    """
    # layer_sq[key] = sum of grad^2 for that layer
    layer_sq: dict[str, torch.Tensor] = {}

    for name, param in model.named_parameters():
        if param.grad is None: continue

        # 解析层
        if name.startswith("model.blocks."):
            parts = name.split(".")
            idx = int(parts[2])  # model.blocks.<idx>.xxx
            key = f"block_{idx:02d}"
        elif name.startswith("model.word_emb."): key = "emb"
        elif name.startswith("model.final_layer_norm."): key = "final_ln"
        else: key = "other"

        # 懒初始化累计器（与梯度在同 device）
        if key not in layer_sq:
            layer_sq[key] = torch.zeros((), device=param.grad.device, dtype=torch.float32)
        # 累加 ||grad||^2（用 float32 统计更稳）
        layer_sq[key] += param.grad.detach().float().pow(2).sum()

    # sqrt(sum) -> L2 norm
    layer_norm = {k: math.sqrt(v.item()) for k, v in layer_sq.items()}
    return layer_norm
