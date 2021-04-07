from logging import getLogger

import torch.nn as nn
import torchvision

__all__ = ["get_model"]

model_names = ["resnet18", "resnet34", "resnet50"]
logger = getLogger(__name__)


def get_model(name: str, n_classes: int, pretrained: bool = True) -> nn.Module:
    name = name.lower()
    if name not in model_names:
        message = (
            "There is no model appropriate to your choice. "
            "You have to choose resnet18, resnet34, resnet50 as a model."
        )
        logger.error(message)
        raise ValueError(message)

    logger.info("{} will be used as a model.".format(name))

    model = getattr(torchvision.models, name)(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features=in_features, out_features=n_classes, bias=True)

    return model
