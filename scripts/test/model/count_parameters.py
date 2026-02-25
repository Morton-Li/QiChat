import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

from src.module.configuration import QiChatConfig
from src.module.model import QiChatForCausalLM


def main():
    use_device = torch.device('meta')
    model_config = QiChatConfig(
        d_model=1024,
        num_hidden_layers=20,
        n_heads=16,
        d_kv=64,
        n_kv_heads=8,
        d_ff=4096,
    )

    with torch.no_grad(), use_device:
        model = QiChatForCausalLM(config=model_config)
        if torch.empty(0).device.type == 'meta': model.tie_weights()
    total_params = round(sum(p.numel() for p in model.parameters()) / 1_000_000, ndigits=2)
    trainable_params = round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000, ndigits=2)
    print(f' - Total Parameters: {total_params} Million')
    print(f' - Trainable Parameters: {trainable_params} Million')


if __name__ == '__main__':
    main()
