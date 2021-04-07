import os
from logging import getLogger
from typing import Optional

import torch.nn as nn

from .class_weight import get_class_weight

__all__ = ["get_criterion"]
logger = getLogger(__name__)


def get_criterion(
    use_class_weight: bool = False,
    train_csv_file: Optional[str] = None,
    device: Optional[str] = None,
) -> nn.Module:

    if use_class_weight:
        if train_csv_file is None or not os.path.exists(train_csv_file):
            message = "the path to a csv file for training is invalid."
            logger.error(message)
            raise FileNotFoundError(message)

        if device is None:
            message = "you should specify a device when you use class weight."
            logger.error(message)
            raise ValueError(message)

        class_weight = get_class_weight(train_csv_file).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weight)
    else:
        criterion = nn.CrossEntropyLoss()

    return criterion
