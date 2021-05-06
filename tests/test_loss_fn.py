import pytest
import torch

from src.libs.loss_fn import get_criterion


def test_get_criterion() -> None:
    with pytest.raises(ValueError):
        get_criterion(True, device="cpu")

    with pytest.raises(ValueError):
        get_criterion(True, dataset_name="flower")

    with pytest.raises(ValueError):
        get_criterion(True, dataset_name="hoge", device="cpu")

    criterion = get_criterion(False)

    pred = torch.rand((2, 10))
    pred = torch.softmax(pred, dim=1)
    gt = torch.tensor([0, 1])

    loss = criterion(pred, gt)
    assert loss > 0
    assert criterion.weight is None

    criterion = get_criterion(True, dataset_name="flower", device="cpu")
    assert criterion.weight is not None
