import random
from logging import getLogger

import torch

logger = getLogger(__name__)


def set_seed(seed: int = 0) -> None:
    # set seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    logger.info("Finished setting up seed.")
