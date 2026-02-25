import importlib
import sys
from pathlib import Path
from typing import Callable

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

from scripts.test.utils import set_seed, generate_random_inputs, get_device, init_model
from src.module import QiChatForCausalLM
from src.tokenizer import QiTianTokenizerFast, get_tokenizer
from src.utils.masks import build_span_mask


class FnHook:
    def __init__(
        self,
        target_fn: str,
        callback_fn: Callable[[...], ...],
        is_passthrough: bool = True,
    ) -> None:
        """
        A simple function hook that intercepts calls to a target function and allows a callback to be executed.
        Args:
            target_fn: The full name of the target function to hook, e.g. 'torch.nn.functional.cross_entropy'.
            callback_fn: A callable that will be called with the same arguments as the target function whenever it is called.
            is_passthrough:
                如果为 True，将在调用目标函数前调用 callback_fn，并且继续调用原函数，最终返回原函数的结果。
                如果为 False，将在调用目标函数前调用 callback_fn，并且不调用原函数，最终返回 callback_fn 的结果。
        """
        self.module, self.fn_name = target_fn.rsplit('.', 1)
        self.module = importlib.import_module(self.module)
        self.orig_fn = getattr(self.module, self.fn_name)
        self.callback_fn = callback_fn
        self.is_passthrough = is_passthrough

    def apply(self, **extra_kwargs):
        """
        Apply the hook by replacing the target function with a wrapper that calls the callback function.
        Args:
            extra_kwargs: Additional keyword arguments to pass to the callback function when it is called.
        """
        def wrap_fn(*args, **kwargs):
            print(f"[FnHook] Intercepted call to {self.module.__name__}.{self.fn_name}")
            result = self.callback_fn(*args, **kwargs, **extra_kwargs)
            if self.is_passthrough: return self.orig_fn(*args, **kwargs)
            return result
        setattr(self.module, self.fn_name, wrap_fn)
    def restore(self): setattr(self.module, self.fn_name, self.orig_fn)


def wrapped_cross_entropy(*args, **kwargs):
    logits = kwargs.get("input", None)
    if logits is None and len(args) >= 1: logits = args[0]
    assert logits is not None, f"Could not find logits in cross_entropy arguments."
    target = kwargs.get("target", None)
    if target is None and len(args) >= 2: target = args[1]
    assert target is not None, f"Could not find labels in cross_entropy arguments."

    orig_input_ids = kwargs.pop("orig_input_ids")
    orig_labels = kwargs.pop("orig_labels")

    print(f"Original input_ids shape: {orig_input_ids.shape}, labels shape: {orig_labels.shape}")
    print(f"Intercepted logits shape: {logits.shape}, labels shape: {target.shape}")
    print(f"Original input_ids:\n{orig_input_ids}")
    print(f"Original labels:\n{orig_labels}")
    print(f"Intercepted labels:\n{target}")


def main():
    set_seed(seed=36)
    use_device: torch.device = get_device(device='auto')
    print(f'Using device: {use_device.type}')

    tokenizer: QiTianTokenizerFast = get_tokenizer()
    model: QiChatForCausalLM = init_model(
        param_size='73M',
        attn_implementation='sdpa',
        dtype='bfloat16',
        use_device=use_device,
    )
    if model.get_input_embeddings().weight.size(0) != len(tokenizer):
        print(f'Token embeddings size ({model.get_input_embeddings().weight.size(0)}) does not match tokenizer size ({len(tokenizer)}). Resizing token embeddings.')
        model.resize_token_embeddings(len(tokenizer))
    model.eval()

    fh = FnHook(
        target_fn='torch.nn.functional.cross_entropy',
        callback_fn=wrapped_cross_entropy,
        is_passthrough=True,
    )

    # PreTrain
    inputs = generate_random_inputs(
        batch_size=2,
        seq_length=8,
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        pad_side='right',
        dtype=model.config.dtype,
        device=use_device,
    )

    fh.apply(orig_input_ids=inputs['input_ids'], orig_labels=inputs['labels'])
    with torch.no_grad():
        _ = model(**inputs)

    # FineTuning
    conversations = [
        [
            {"role": "user", "content": "请解释一下量子纠缠是什么？"},
            {"role": "assistant", "content": "量子纠缠是一种发生在多个粒子之间的量子关联现象，其特点是无论粒子之间的距离有多远，对其中一个粒子的测量都会立即影响到另一个粒子的状态。"}
        ],
        [
            {"role": "user", "content": "美国独立战争是什么时候发生的？"},
            {"role": "assistant", "content": "美国独立战争发生在1775年至1783年之间，是美国十三个殖民地为了摆脱英国的统治而进行的一场革命战争。最终，美国赢得了独立，成立了美利坚合众国。"}
        ]
    ]
    pad_token_id = tokenizer.pad_token_id
    input_ids = tokenizer.apply_chat_template(
        conversation=conversations,
        tokenize=True,
        padding=True,
        add_generation_prompt=False,
        return_tensors='pt',
    ).input_ids[:, 2:]  # Remove BOS token

    label_ids = input_ids.clone()
    ignore_mask = ~build_span_mask(
        input_ids=label_ids,
        start_token_id=tokenizer.convert_tokens_to_ids('<|assistant|>'),
        end_token_id=tokenizer.convert_tokens_to_ids('<|eot|>'),
        include_start_token=False,
        include_end_token=True,
    )
    label_ids[ignore_mask] = -100
    label_ids[label_ids == pad_token_id] = -100

    inputs = {
        'input_ids': input_ids.to(device=use_device),
        'attention_mask': (input_ids != pad_token_id).to(device=use_device),  # 生成注意力掩码
        'labels': label_ids.to(device=use_device),
        'loss_ignore_index': -100,
    }
    fh.apply(orig_input_ids=inputs['input_ids'], orig_labels=inputs['labels'])
    with torch.no_grad():
        _ = model(**inputs)


if __name__ == '__main__':
    main()
