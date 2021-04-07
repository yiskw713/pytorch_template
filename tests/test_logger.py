import os
from logging import DEBUG, INFO
from typing import Tuple, Union

import pytest
from _pytest.logging import LogCaptureFixture

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


def test_update(epoch_result: Tuple[Union[int, float], ...], caplog: LogCaptureFixture):
    caplog.set_level(DEBUG)

    log_path = "./tests/tmp/log.csv"
    logger = TrainLogger(log_path, False)
    logger.update(*epoch_result)

    # test logs
    assert (
        "src.libs.logger",
        DEBUG,
        "training logs are saved.",
    ) in caplog.record_tuples

    assert (
        "src.libs.logger",
        INFO,
        f"epoch: {epoch_result[0]}\t"
        f"epoch time[sec]: {epoch_result[2] + epoch_result[6]}\t"
        f"lr: {epoch_result[1]}\t"
        f"train loss: {epoch_result[3]:.4f}\tval loss: {epoch_result[7]:.4f}\t"
        f"val_acc1: {epoch_result[8]:.5f}\tval_f1s: {epoch_result[9]:.5f}",
    ) in caplog.record_tuples

    assert os.path.exists(log_path)
    assert len(logger.df) == 1

    logger.update(*epoch_result)
    assert len(logger.df) == 2


def test_init_load(caplog: LogCaptureFixture):
    caplog.set_level(DEBUG)

    log_path = "./tests/tmp/no_exist_log.csv"

    with pytest.raises(FileNotFoundError):
        logger = TrainLogger(log_path, True)

    log_path = "./tests/tmp/log.csv"
    logger = TrainLogger(log_path, True)

    # test logs
    assert (
        "src.libs.logger",
        INFO,
        "successfully loaded log csv file.",
    ) in caplog.record_tuples

    assert len(logger.df) == 2
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
