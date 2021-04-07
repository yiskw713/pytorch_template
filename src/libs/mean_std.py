from logging import getLogger
from typing import List

logger = getLogger(__name__)


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
