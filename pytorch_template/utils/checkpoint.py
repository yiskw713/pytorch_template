from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger


def save_checkpoint(
    save_path: str,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    best_loss: float,
) -> None:
    # TODO: save path
    # TODO: kwargs
    save_states = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_loss": best_loss,
    }

    torch.save(save_states, save_path)
    logger.debug(f"successfully saved the ckeckpoint in {save_path}.")


def resume(
    resume_path: str, model: nn.Module, optimizer: optim.Optimizer
) -> Tuple[int, nn.Module, optim.Optimizer, float]:
    try:
        checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage)
        logger.info("loading checkpoint {}".format(resume_path))
    except FileNotFoundError as e:
        logger.exception(f"{e}")

    begin_epoch = checkpoint["epoch"]
    best_loss = checkpoint["best_loss"]
    model.load_state_dict(checkpoint["state_dict"])

    optimizer.load_state_dict(checkpoint["optimizer"])

    logger.info("training will start from {} epoch".format(begin_epoch))

    return begin_epoch, model, optimizer, best_loss
