import os

from transformers import PreTrainedTokenizerFast

from .utils.path import data_path


class QiTianTokenizerFast(PreTrainedTokenizerFast):
    """
    QiTianTokenizerFast
    https://huggingface.co/Morton-Li/QiTianTokenizer-Base
    """
    model_input_names: list[str] = ["input_ids", "attention_mask"]


def get_tokenizer() -> QiTianTokenizerFast:
    """Get the QiTianTokenizerFast tokenizer."""
    model_path = data_path('tokenizer', 'model')
    file_list = [file for file in os.listdir(model_path)]
    if not file_list or not all(
        required_file in file_list
        for required_file in ['tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja']
    ):
        raise FileNotFoundError('Tokenizer files are missing in the specified model path.')
    return QiTianTokenizerFast.from_pretrained(pretrained_model_name_or_path=model_path)
