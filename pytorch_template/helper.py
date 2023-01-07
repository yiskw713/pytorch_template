import time
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader

from .utils import AverageMeter, ProgressMeter
from .metric import calc_accuracy

__all__ = ["train", "evaluate"]


def train(
    loader: DataLoader,
    model: nn.Module,
    criterion: Any,
    optimizer: optim.Optimizer,
    epoch: int,
    device: str,
    interval_of_progress: int = 50,
    topk: Tuple[int, ...] = (1, 5),
) -> Dict[str, Any]:
    batch_time_meter = AverageMeter("Time", ":6.3f")
    data_time_meter = AverageMeter("Data", ":6.3f")
    loss_meter = AverageMeter("Loss", ":.4e")
    acc_meters = [AverageMeter(f"Acc@{k}", ":6.2f") for k in topk]
    progress = ProgressMeter(
        len(loader),
        [batch_time_meter, data_time_meter, loss_meter] + acc_meters,
        prefix="Epoch: [{}]".format(epoch),
    )

    # keep predicted results and gts for calculate F1 Score
    gts = []
    preds = []

    # switch to train mode
    model.train()

    end = time.time()
    for i, sample in enumerate(loader):
        # measure data loading time
        data_time_meter.update(time.time() - end)

        x = sample["img"].to(device)
        t = sample["class_id"].to(device)
        batch_size = x.shape[0]

        # compute output and loss
        output = model(x)
        loss = criterion(output, t)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        accs = calc_accuracy(output, t, topk=topk)

        # keep predicted results and gts for calculate F1 Score
        _, pred = output.max(dim=1)
        gt = t.to("cpu").numpy()
        pred = pred.to("cpu").numpy()

        # update meters
        loss_meter.update(loss, batch_size)
        for acc, meter in zip(accs, acc_meters):
            meter.update(acc, batch_size)

        # save the ground truths and predictions in lists
        gts += list(gt)
        preds += list(pred)

        # measure elapsed time
        batch_time_meter.update(time.time() - end)
        end = time.time()

        # show progress bar per 50 iteration
        if i != 0 and i % interval_of_progress == 0:
            progress.display(i)

    # calculate F1 Score
    f1s = f1_score(gts, preds, average="macro")

    res = {"Train loss": loss_meter.get_average(), "Train F1 score": f1s}
    res.update(zip([f"Train Acc@{k}" for k in topk], acc_meters))
    return res


@torch.no_grad
def evaluate(
    loader: DataLoader,
    model: nn.Module,
    criterion: Any,
    device: str,
    topk: Tuple[int, ...] = (1, 5),
) -> Tuple[Dict[str, Any], np.ndarray]:
    loss_meter = AverageMeter("Loss", ":.4e")
    acc_meters = [AverageMeter(f"Acc@{k}", ":6.2f") for k in topk]

    # keep predicted results and gts for calculate F1 Score
    gts = []
    preds = []

    # calculate confusion matrix
    n_classes = loader.dataset.get_n_classes()
    c_matrix = np.zeros((n_classes, n_classes), dtype=np.int32)

    # switch to evaluate mode
    model.eval()

    for sample in loader:
        x = sample["img"].to(device)
        t = sample["class_id"].to(device)
        batch_size = x.shape[0]

        # compute output and loss
        output = model(x)
        loss = criterion(output, t)

        # measure accuracy and record loss
        accs = calc_accuracy(output, t, topk=(1,))

        # keep predicted results and gts for calculate F1 Score
        _, pred = output.max(dim=1)
        gt = t.to("cpu").numpy()
        pred = pred.to("cpu").numpy()

        # update meters
        loss_meter.update(loss, batch_size)
        for acc, meter in zip(accs, acc_meters):
            meter.update(acc, batch_size)

        # keep predicted results and gts for calculate F1 Score
        gts += list(gt)
        preds += list(pred)

        c_matrix += confusion_matrix(
            gt,
            pred,
            labels=[i for i in range(n_classes)],
        )

    f1s = f1_score(gts, preds, average="macro")

    res = {"Evaluation loss": loss_meter.get_average(), "Evaluation F1 score": f1s}
    res.update(zip([f"Evaluation Acc@{k}" for k in topk], acc_meters))

    return res, c_matrix
