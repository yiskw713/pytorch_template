import torch.optim as optim

from ..utils.factory import Factory


class OptimizerFactory(Factory[optim.Optimizer]):
    pass


# Register all optimizers defined in torch.optim
OptimizerFactory.register_from_module(optim)
