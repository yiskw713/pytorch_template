# 今回は torchvision で提供されているものを用います．
# TODO: imgaug
from typing import List

from loguru import logger
from torchvision.transforms import (
    ColorJitter,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    ToTensor,
)


def get_mean(norm_value: float = 255) -> List[float]:
    # mean of imagenet
    mean = [123.675 / norm_value, 116.28 / norm_value, 103.53 / norm_value]

    logger.info(f"mean value: {mean}")
    return mean


def get_std(norm_value: float = 255) -> List[float]:
    # std fo imagenet
    std = [58.395 / norm_value, 57.12 / norm_value, 57.375 / norm_value]

    logger.info(f"std value: {std}")
    return std


def get_train_transform(height: int, width: int) -> Compose:
    return Compose(
        [
            RandomResizedCrop(size=(height, width)),
            RandomHorizontalFlip(),
            ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            ToTensor(),
            Normalize(mean=get_mean(), std=get_std()),
        ]
    )


def get_test_transform() -> Compose:
    return Compose([ToTensor(), Normalize(mean=get_mean(), std=get_std())])
