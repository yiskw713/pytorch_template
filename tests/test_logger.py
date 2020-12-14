import os
from typing import Tuple, Union

import pytest
from _pytest.capture import CaptureFixture

from src.libs.logger import TrainLogger


@pytest.fixture()
def epoch_result() -> Tuple[Union[int, float], ...]:
    results = (
        0,
        0.1,
        20,
        0.05,
        30.0,
        30.0,
        10,
        0.03,
        28.3,
        28.3,
    )
    return results


def test_update(epoch_result: Tuple[int], capfd: CaptureFixture):
    log_path = "./tests/tmp/log.csv"
    logger = TrainLogger(log_path, False)
    logger.update(*epoch_result)

    # test printed string
    _, err = capfd.readouterr()
    assert err == ""

    assert os.path.exists(log_path)
    assert len(logger.df) == 1


def test_init_load():
    log_path = "./tests/tmp/no_exist_log.csv"

    with pytest.raises(FileNotFoundError):
        logger = TrainLogger(log_path, True)

    log_path = "./tests/tmp/log.csv"
    logger = TrainLogger(log_path, True)

    assert len(logger.df) == 1
    assert logger.df.iloc[0]["epoch"] == 0
    assert logger.df.iloc[0]["lr"] == 0.1
    assert logger.df.iloc[0]["train_time[sec]"] == 20
    assert logger.df.iloc[0]["train_loss"] == 0.05
    assert logger.df.iloc[0]["train_acc@1"] == 30.0
    assert logger.df.iloc[0]["train_f1s"] == 30.0
    assert logger.df.iloc[0]["val_time[sec]"] == 10
    assert logger.df.iloc[0]["val_loss"] == 0.03
    assert logger.df.iloc[0]["val_acc@1"] == 28.3
    assert logger.df.iloc[0]["val_f1s"] == 28.3

    os.remove(log_path)
