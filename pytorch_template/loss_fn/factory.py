import torch.nn as nn

from ..utils.factory import Factory


class LossFactory(Factory[nn.Module]):
    pass
