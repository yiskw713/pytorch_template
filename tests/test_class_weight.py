from typing import Tuple

import pandas as pd
import pytest
import torch

from src.libs.loss_fn.class_weight import get_class_num, get_class_weight


@pytest.fixture()
def train_data_csv() -> Tuple[pd.DataFrame, str]:
    csv_file = "./src/csv/train.csv"
    df = pd.read_csv(csv_file)
    return df, csv_file


def test_get_class_num(train_data_csv: Tuple[pd.DataFrame, str]) -> None:
    train_data, csv_file = train_data_csv

    class_num = get_class_num(csv_file)

    assert class_num.shape == (5,)
    assert len(train_data) == class_num.sum().item()
    assert torch.all(class_num > 0)


def test_get_class_weight(train_data_csv: Tuple[pd.DataFrame, str]) -> None:
    _, csv_file = train_data_csv

    class_weight = get_class_weight(csv_file)

    assert class_weight.shape == (5,)
    assert torch.all(class_weight > 0)
    assert class_weight.dtype == torch.float
