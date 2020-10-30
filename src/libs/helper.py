import time
from typing import Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader

from libs.meter import AverageMeter, ProgressMeter
from libs.metric import calc_accuracy


def train(
    train_loader: DataLoader,
    model: nn.Module,
    criterion: Any,
    optimizer: optim.Optimizer,
    epoch: int,
    device: str,
) -> Tuple[float, float, float]:
    # 平均を計算してくれるクラス
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")

    # 進捗状況を表示してくれるクラス
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch),
    )

    # keep predicted results and gts for calculate F1 Score
    gts = []
    preds = []

    # switch to train mode
    model.train()

    end = time.time()
    for i, sample in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        x = sample["img"].to(device)
        t = sample["class_id"].to(device)

        batch_size = x.shape[0]

        # compute output and loss
        output = model(x)
        loss = criterion(output, t)

        # measure accuracy and record loss
        accs = calc_accuracy(output, t, topk=(1,))
        acc1 = accs[0]

        losses.update(loss.item(), batch_size)
        top1.update(acc1, batch_size)

        # keep predicted results and gts for calculate F1 Score
        _, pred = output.max(dim=1)
        gts += list(t.to("cpu").numpy())
        preds += list(pred.to("cpu").numpy())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # show progress bar per 50 iteration
        if i != 0 and i % 50 == 0:
            progress.display(i)

    # calculate F1 Score
    f1s = f1_score(gts, preds, average="macro")

    return losses.get_average(), top1.get_average(), f1s


def validate(
    val_loader: DataLoader, model: nn.Module, criterion: Any, device: str
) -> Tuple[float, float, float]:
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")

    # keep predicted results and gts for calculate F1 Score
    gts = []
    preds = []

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for sample in val_loader:
            x = sample["img"].to(device)
            t = sample["class_id"].to(device)

            batch_size = x.shape[0]

            # compute output and loss
            output = model(x)
            loss = criterion(output, t)

            # measure accuracy and record loss
            accs = calc_accuracy(output, t, topk=(1,))
            acc1 = accs[0]

            losses.update(loss.item(), batch_size)
            top1.update(acc1, batch_size)

            # keep predicted results and gts for calculate F1 Score
            _, pred = output.max(dim=1)
            gts += list(t.to("cpu").numpy())
            preds += list(pred.to("cpu").numpy())

    f1s = f1_score(gts, preds, average="macro")

    return losses.get_average(), top1.get_average(), f1s


def evaluate(
    loader: DataLoader, model: nn.Module, n_classes: int, device: str
) -> Tuple[float, float, List[List[int]]]:
    top1 = AverageMeter("Acc@1", ":6.2f")

    # keep predicted results and gts for calculate F1 Score
    gts = []
    preds = []

    # calculate confusion matrix
    c_matrix = np.zeros((n_classes, n_classes), dtype=np.int)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, sample in enumerate(loader):
            x = sample["img"].to(device)
            t = sample["class_id"].to(device)

            batch_size = x.shape[0]

            # compute output and loss
            output = model(x)

            # measure accuracy and record loss
            accs = calc_accuracy(output, t, topk=(1,))
            acc1 = accs[0]
            top1.update(acc1, batch_size)

            # keep predicted results and gts for calculate F1 Score
            _, pred = output.max(dim=1)
            gts += list(t.to("cpu").numpy())
            preds += list(pred.to("cpu").numpy())

            c_matrix += confusion_matrix(
                t.to("cpu").numpy(),
                pred.to("cpu").numpy(),
                labels=[i for i in range(n_classes)],
            )

    f1s = f1_score(gts, preds, average="macro")

    return top1.get_average(), f1s, c_matrix
