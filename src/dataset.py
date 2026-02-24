import numpy
import torch
from torch.utils.data import Dataset


class QiDataset(Dataset):
    def __init__(
        self,
        input_ids: list[numpy.ndarray] | list[list[int]],
        label_ids: list[numpy.ndarray] | list[list[int]],
    ):
        super().__init__()

        if len(input_ids) != len(label_ids):
            raise ValueError('Input IDs and label IDs must have the same length.')

        self.input_ids: list[numpy.ndarray] | list[list[int]] = input_ids
        self.label_ids: list[numpy.ndarray] | list[list[int]] = label_ids

    def __len__(self) -> int:
        """
        Get the length of the dataset
        :return:
        """
        return len(self.input_ids)

    def __getitem__(self, index: int) -> tuple[torch.LongTensor, torch.LongTensor]:
        """
        Get the item of the dataset
        :param index: Index
        :return:
        """
        input_tensor = torch.as_tensor(self.input_ids[index].copy(), dtype=torch.long)
        label_tensor = torch.as_tensor(self.label_ids[index].copy(), dtype=torch.long)
        return input_tensor, label_tensor

    def random_sample(self, n: int) -> tuple[list[torch.LongTensor], list[torch.LongTensor]]:
        """
        Random sample
        :param n: Sample size
        :return:
        """
        data_len = len(self.input_ids)

        if data_len == 0:
            raise ValueError('The dataset is empty.')

        if n > data_len:
            raise ValueError('The sample size is greater than the dataset size.')

        indices = numpy.random.choice(data_len, size=n, replace=False).tolist()

        sampled_input_ids = [torch.as_tensor(self.input_ids[i].copy(), dtype=torch.long) for i in indices]
        sampled_label_ids = [torch.as_tensor(self.label_ids[i].copy(), dtype=torch.long) for i in indices]

        return sampled_input_ids, sampled_label_ids
