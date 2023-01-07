from typing import Optional

import torch.nn as nn
from loguru import logger

from ..dataset_csv import DATASET_CSVS
from .class_weight import get_class_weight

__all__ = ["get_criterion"]


def get_criterion(
    use_class_weight: bool = False,
    dataset_name: Optional[str] = None,
    device: Optional[str] = None,
) -> nn.Module:

    if use_class_weight:
        if dataset_name is None:
            message = "dataset_name used for training should be specified."
            logger.error(message)
            raise ValueError(message)

        if device is None:
            message = "you should specify a device when you use class weight."
            logger.error(message)
            raise ValueError(message)

        if dataset_name not in DATASET_CSVS:
            message = "dataset_name is invalid."
            logger.error(message)
            raise ValueError(message)

        train_csv_file = DATASET_CSVS[dataset_name].train
        class_weight = get_class_weight(train_csv_file).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weight)
    else:
        criterion = nn.CrossEntropyLoss()

    return criterion
