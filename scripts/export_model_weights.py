import argparse
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.path import data_path


def main(
    pth_name: str,
    save_path: Path | str,
):
    print('Loading model weights...')
    model_weights = torch.load(data_path('checkpoint', pth_name if pth_name.endswith('.pth') else f'{pth_name}.pth'))
    model_weights = model_weights['state']['weights']
    if isinstance(save_path, str): save_path = Path(save_path)
    print(f'Saving model weights to {save_path} ...')
    torch.save(model_weights, save_path)
    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Export model weights from a .pth file."
    )
    parser.add_argument(
        "--pth_name",
        default='model_weights.pth',
        help="Name of the .pth file containing model weights."
    )
    parser.add_argument(
        "--save_path",
        default=data_path('checkpoint', 'exported_model_weights.pt'),
        help="Path to save the exported model weights."
    )
    args = parser.parse_args()

    main(
        pth_name=args.pth_name,
        save_path=args.save_path
    )
