import pytest
import torch

from src.libs.models import get_model


def test_get_model() -> None:
    with pytest.raises(ValueError):
        get_model("modelname", 10, False)

    model = get_model("resnet18", 10)

    x = torch.rand((2, 3, 112, 112))
    y = model(x)

    assert y.shape == (2, 10)
