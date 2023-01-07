import torchvision
from torch.utils.data import Dataset

from ..utils.factory import Factory


class DatasetFactory(Factory[Dataset]):
    pass


# Register Dataset class defined in torchvision.
DatasetFactory.register_from_module(torchvision.datasets)
