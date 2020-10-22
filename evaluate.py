import argparse
import csv
import os
import sys

import pandas as pd
import torch
import yaml
from torchvision.transforms import Compose, Normalize, ToTensor

from libs.class_label_map import get_cls2id_map
from libs.dataset import get_dataloader
from libs.mean import get_mean, get_std
from libs.models import get_model
from libs.config import Config
from libs.helper import evaluate


def get_arguments() -> argparse.Namespace:
    """
    parse all the arguments from command line inteface
    return a list of parsed arguments
    """

    parser = argparse.ArgumentParser(
        description="train a network for image classification with Flowers Recognition Dataset"
    )
    parser.add_argument("config", type=str, help="path of a config file")
    parser.add_argument("mode", type=str, help="validation or test")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="""path to the trained model. If you do not specify, the trained model,
            'best_acc1_model.prm' in result directory will be used.""",
    )

    return parser.parse_args()


def main() -> None:
    args = get_arguments()

    # configuration
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
    CONFIG = Config(**config_dict)
    result_path = os.path.dirname(args.config)

    # cpu or cuda
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    else:
        print("You have to use GPUs because training CNN is computationally expensive.")
        sys.exit(1)

    # Dataloader
    assert args.mode in ["validation", "test"]

    transform = Compose([ToTensor(), Normalize(mean=get_mean(), std=get_std())])

    loader = get_dataloader(
        CONFIG.val_csv if args.mode == "validation" else CONFIG.test_csv,
        batch_size=1,
        shuffle=False,
        num_workers=CONFIG.num_workers,
        pin_memory=True,
        transform=transform,
    )

    # the number of classes
    n_classes = len(get_cls2id_map())

    model = get_model(CONFIG.model, n_classes, pretrained=CONFIG.pretrained)

    # send the model to cuda/cpu
    model.to(device)

    # load the state dict of the model
    if args.model is not None:
        state_dict = torch.load(args.model)
    else:
        state_dict = torch.load(os.path.join(result_path, "best_model.prm"))

    model.load_state_dict(state_dict)

    # train and validate model
    print("\n------------------------Start testing------------------------\n")

    # evaluation
    acc1, f1s, c_matrix = evaluate(loader, model, n_classes, device)

    print("acc1: {:.5f}\tF1 Score: {:.5f}".format(acc1, f1s))

    df = pd.DataFrame(
        {"acc@1": [acc1], "f1score": [f1s]}, columns=["acc@1", "f1score"], index=None
    )

    df.to_csv(os.path.join(result_path, "{}_log.csv").format(args.mode), index=False)

    with open(
        os.path.join(result_path, "{}_c_matrix.csv").format(args.mode), "w"
    ) as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerows(c_matrix)

    print("Done.")


if __name__ == "__main__":
    main()
