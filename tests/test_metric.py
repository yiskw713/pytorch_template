import pytest
import torch

from src.libs.metric import calc_accuracy


@pytest.fixture()
def predictions() -> torch.Tensor:
    # shape (N, C) = (5, 5)
    preds = torch.tensor(
        [
            [0.05, 0.1, 0.15, 0.2, 0.5],
            [0.05, 0.1, 0.15, 0.2, 0.5],
            [0.05, 0.1, 0.15, 0.2, 0.5],
            [0.05, 0.1, 0.15, 0.2, 0.5],
            [0.05, 0.1, 0.15, 0.2, 0.5],
        ]
    )
    return preds


@pytest.fixture()
def ground_truths() -> torch.Tensor:
    # shape (N, ) = (1, )
    gts = torch.tensor([4, 3, 2, 1, 0])
    return gts


def test_calc_accuracy(predictions: torch.Tensor, ground_truths: torch.Tensor) -> None:
    top1, top2, top3, top4, top5 = calc_accuracy(
        predictions, ground_truths, topk=(1, 2, 3, 4, 5)
    )
    assert top1 == 1 / 5 * 100
    assert top2 == 2 / 5 * 100
    assert top3 == 3 / 5 * 100
    assert top4 == 4 / 5 * 100
    assert top5 == 5 / 5 * 100
