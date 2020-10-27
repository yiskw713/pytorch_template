import sys

import torch


def get_device(allow_only_gpu: bool = True):
    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = True
    else:
        if allow_only_gpu:
            print("You should use GPUs for training CNNs.")
            sys.exit(0)

        device = "cpu"
        print(
            """CPU will be used for training. It is better to use GPUs instead
            because training CNN is computationally expensive.
            """
        )

    return device
