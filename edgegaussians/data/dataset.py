import torch

from torch import Tensor
from torch.utils.data import Dataset
from typing import Dict

from edgegaussians.data.dataparsers import DataParser


class InputDataset(Dataset):
    """Dataset that returns images.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs
    """

    def __init__(self, dataparser: DataParser):
        super().__init__()
        self.views = dataparser.views

    def __len__(self):
        return len(self.views)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """Returns the image and mask of the idx-th sample.

        Args:
            idx: The sample index in the dataset.
        """
        image = self.views[idx]['image']
        return {'image':image, 'idx':torch.tensor(idx, dtype=torch.int64)}