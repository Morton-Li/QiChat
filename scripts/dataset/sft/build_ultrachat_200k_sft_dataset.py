import sys
from pathlib import Path

from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

from src.utils.path import data_path
from scripts.dataset.utils import stream_corpus_batches, build_sft_dataset


def main():
    print('Building UltraChat 200k SFT dataset...')

    conversations = []
    for file in tqdm(
        [
            'train_sft-00000-of-00003-a3ecf92756993583.parquet',
            'train_sft-00001-of-00003-0a1804bcb6ae68c6.parquet',
            'train_sft-00002-of-00003-ee46ed25cfae92c6.parquet',
        ],
        desc='Processing corpus files',
        unit='file',
        dynamic_ncols=True,
    ):
        for batch_df in stream_corpus_batches(
            parquet_path=data_path('tokenizer', 'source_corpus', 'HuggingFaceH4-ultrachat_200k', file),
            batch_size=1024,
            text_field='messages'
        ):
            for message in batch_df.dropna().reset_index(drop=True)['messages'].tolist():
                # message 为 ndarray
                conversations.append(message.tolist())
    print(f'Total dialogues loaded: {len(conversations)}')

    build_sft_dataset(
        dataset_name='ultrachat_200k',
        conversations=conversations,
        add_system_block=False,
        assistant_token='<|assistant|>',
        end_of_turn_token='<|eot|>',
    )
    print('Dialogue dataset built successfully.')


if __name__ == '__main__':
    main()
