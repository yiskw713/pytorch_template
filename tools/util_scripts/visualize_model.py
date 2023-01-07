import argparse
import os
import sys

import hiddenlayer as hl
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from libs.models import get_model


def get_arguments() -> argparse.Namespace:
    """parse all the arguments from command line inteface return a list of
    parsed arguments."""

    parser = argparse.ArgumentParser(description="Model visualization.")
    parser.add_argument(
        "model",
        type=str,
        choices=["resnet18", "resnet34", "resnet50"],
        help="name of the model you want to visualize.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./imgs",
        help="a directory where images will be saved",
    )

    return parser.parse_args()


def main() -> None:
    args = get_arguments()

    model = get_model(args.model, 10)
    save_path = os.path.join(args.save_dir, f"{args.model}.png")

    hl_graph = hl.build_graph(model, torch.zeros([1, 3, 224, 224]))
    hl_graph.save(save_path, format="png")


if __name__ == "__main__":
    main()
