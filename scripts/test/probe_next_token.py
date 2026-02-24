import sys
from pathlib import Path
from typing import Literal, Optional

import pandas
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import get_config
from src.module.model import QiChatForCausalLM
from src.tokenizer import QiTianTokenizerFast, get_tokenizer
from src.utils.type import DotDict
from scripts.test.utils import set_seed, get_device, init_model, load_model_checkpoint


def main(
    target_token_ids: Optional[list[int]] = None,
    seed: int = 36,
    use_device: Literal['auto', 'cuda', 'mps', 'cpu'] = 'auto'
):
    print(f'Random seed set to: {seed}')
    set_seed(seed=seed)

    use_device: torch.device = get_device(device=use_device)
    print(f'Using device: {use_device.type}')

    tokenizer: QiTianTokenizerFast = get_tokenizer()
    input_text_pt = tokenizer.apply_chat_template(
        conversation=[
            {"role": 'user', "content": '相对论的作者是谁？'},
        ],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors='pt'
    )

    config: DotDict = get_config()
    model: QiChatForCausalLM = init_model(
        param_size=config.model.param_size,
        attn_implementation=config.model.attn_implementation,
        dtype=config.model.dtype,
        additional_model_config_kwargs=config.model.model_config_kwargs,
        checkpoint=load_model_checkpoint(map_location=use_device)['state']['weights'],
        use_device=use_device,
    )
    model.eval()

    with torch.no_grad():
        logits = model(input_ids=input_text_pt.to(use_device)).logits[:, -1]  # 下一token
        logits = logits[0]  # (vocab_size,)
        topk = torch.topk(logits, k=30).indices.tolist()
        probs = torch.softmax(logits, dim=-1)

    if target_token_ids is not None:
        print(f'Target Token Predictions:')
        print(
            pandas.DataFrame({
                'Target Token IDs': target_token_ids,
                'Tokens': [tokenizer.decode([idx]) for idx in target_token_ids],
                'Logits': [logits[idx].item() for idx in target_token_ids],
                'Probabilities': [probs[idx].item() for idx in target_token_ids],
                'Ranks': [int((probs > probs[idx]).sum().item()) + 1 for idx in target_token_ids],
            }).to_string()
        )

    print(f'Top-30 Predictions:')
    print(
        pandas.DataFrame({
            'IDs': topk,
            'Tokens': [tokenizer.decode([idx]) for idx in topk],
            'Logits': [logits[idx].item() for idx in topk],
            'Probabilities': [probs[idx].item() for idx in topk],
        }).to_string()
    )


if __name__ == '__main__':
    main()
