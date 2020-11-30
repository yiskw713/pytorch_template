import torch


def get_device(allow_only_gpu: bool = True) -> str:
    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = True
    else:
        if allow_only_gpu:
            raise ValueError(
                """You can use only cpu while you don't allow the use of cpu alone during training.
                """
            )

        device = "cpu"
        print(
            """CPU will be used for training. It is better to use GPUs instead
            because training CNN is computationally expensive.
            """
        )

    return device
