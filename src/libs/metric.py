from typing import List, Tuple

import torch


def calc_accuracy(
    output: torch.Tensor, target: torch.Tensor, topk: Tuple[int] = (1,)
) -> List[float]:
    """Computes the accuracy over the k top predictions for the specifiedvalues
    of k."""

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1)
            correct_k = correct_k.float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res
