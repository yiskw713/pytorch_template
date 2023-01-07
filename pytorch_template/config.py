from pprint import pformat
from typing import Any, Dict, Tuple
from pydantic import BaseModel

import yaml
from loguru import logger

__all__ = ["get_config"]


class ModelConfig(BaseModel):
    name: str
    pretrained: bool


class LossConfig(BaseModel):
    name: str
    # whether you use class weight to calculate cross entropy or not
    use_class_weight: bool


class DatasetConfig(BaseModel):
    name: str
    train_csv: str
    val_csv: str
    test_csv: str

    batch_size: int

    width: int
    height: int


class Config(BaseModel):
    """Experimental configuration class."""

    model: ModelConfig
    loss: LossConfig
    dataset: DatasetConfig

    n_epochs: int = 50
    learning_rate: float = 0.003

    # TODO:
    def __post_init__(self) -> None:
        logger.info(
            "TODO"
            # "Experiment Configuration\n" + pformat(dataclasses.asdict(self), width=1)
        )


def get_config(config_path: str) -> Config:
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    config = Config(**config_dict)
    return config
