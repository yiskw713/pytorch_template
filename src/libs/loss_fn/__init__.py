import os
from typing import Optional

import torch.nn as nn

from .class_weight import get_class_weight

__all__ = ["get_criterion"]


def get_criterion(
    use_class_weight: bool = False,
    train_csv_file: Optional[str] = None,
    device: Optional[str] = None,
) -> nn.Module:

    if use_class_weight:
        if train_csv_file is None or not os.path.exists(train_csv_file):
            raise FileNotFoundError("the path to a csv file for training is invalid.")

        if device is None:
            raise ValueError("you should specify a device when you use class weight.")

        class_weight = get_class_weight(train_csv_file).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weight)
    else:
        criterion = nn.CrossEntropyLoss()

    return criterion
