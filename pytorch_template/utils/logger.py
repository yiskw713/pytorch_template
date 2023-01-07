import sys
from typing import Any

import pandas as pd
import torch
from loguru import logger


class MetricLogger(object):
    def __init__(self, log_path: str, resume: bool, deliminator: str = "\t") -> None:
        self.log_path = log_path
        self.deliminator = deliminator

        if resume:
            self.df = self._load_log(self.log_path)
        else:
            self.df = pd.DataFrame()

    def _load_log(self, log_path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(log_path)
            logger.info("successfully loaded log csv file.")
            return df
        except FileNotFoundError as err:
            logger.exception(f"{err}")
            raise err

    def save_log(self) -> None:
        self.df.to_csv(self.log_path, index=False)
        logger.debug("training logs are saved.")

    def update(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
                kwargs[k] = v

        tmp = pd.Series(**kwargs)

        self.df = self.df.append(tmp, ignore_index=True)

    def print_latest_log(self) -> None:
        res = []
        log_dict = self.df.iloc[-1].to_dict()
        for k, v in log_dict.items():
            res.append(f"{k}: {v}")

        logger.info(self.deliminator.join(res))


def setup_logger(
    log_path: str, log_level: str = "INFO", supress_stdout: bool = False
) -> None:
    # TOOD: use rich
    # from rich import print
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    logger.remove()
    logger.add(log_path, rotation="0:00", format=log_format, level=log_level)

    if supress_stdout:
        logger.add(sys.stdout, format=log_format, level=log_level)
