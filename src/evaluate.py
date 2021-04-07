import argparse
import csv
import datetime
import os
from logging import DEBUG, INFO, basicConfig, getLogger

import pandas as pd
import torch
from torchvision.transforms import Compose, Normalize, ToTensor

from libs.class_id_map import get_cls2id_map
from libs.config import get_config
from libs.dataset import get_dataloader
from libs.device import get_device
from libs.helper import evaluate
from libs.loss_fn import get_criterion
from libs.mean_std import get_mean, get_std
from libs.models import get_model

logger = getLogger(__name__)


def get_arguments() -> argparse.Namespace:
    """parse all the arguments from command line inteface return a list of
    parsed arguments."""

    parser = argparse.ArgumentParser(
        description="""
                    train a network for image classification
                    with Flowers Recognition Dataset
                    """
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Add --debug option if you want to see debug-level logs.",
    )

    return parser.parse_args()


def main() -> None:
    args = get_arguments()

    # configuration
    config = get_config(args.config)
    result_path = os.path.dirname(args.config)

    if args.mode not in ["validation", "test"]:
        message = "args.mode is invalid. ['validation', 'test']"
        logger.error(message)
        raise ValueError(message)

    # setting logger configuration
    logname = os.path.join(
        result_path, f"{datetime.datetime.now():%Y-%m-%d}_{args.mode}.log"
    )
    basicConfig(
        level=DEBUG if args.debug else INFO,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=logname,
    )

    # cpu or cuda
    device = get_device(allow_only_gpu=True)

    # Dataloader
    transform = Compose([ToTensor(), Normalize(mean=get_mean(), std=get_std())])

    loader = get_dataloader(
        config.val_csv if args.mode == "validation" else config.test_csv,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        transform=transform,
    )

    # the number of classes
    n_classes = len(get_cls2id_map())

    model = get_model(config.model, n_classes, pretrained=config.pretrained)

    # send the model to cuda/cpu
    model.to(device)

    # load the state dict of the model
    if args.model is not None:
        state_dict = torch.load(args.model)
    else:
        state_dict = torch.load(os.path.join(result_path, "best_model.prm"))

    model.load_state_dict(state_dict)

    # criterion for loss
    criterion = get_criterion(config.use_class_weight, config.train_csv, device)

    # train and validate model
    logger.info(f"---------- Start evaluation for {args.mode} data ----------")

    # evaluation
    loss, acc1, f1s, c_matrix = evaluate(loader, model, criterion, device)

    logger.info("loss: {:.5f}\tacc1: {:.2f}\tF1 Score: {:.2f}".format(loss, acc1, f1s))

    df = pd.DataFrame(
        {"loss": [loss], "acc@1": [acc1], "f1score": [f1s]},
        columns=["loss", "acc@1", "f1score"],
        index=None,
    )

    df.to_csv(os.path.join(result_path, "{}_log.csv").format(args.mode), index=False)

    with open(
        os.path.join(result_path, "{}_c_matrix.csv").format(args.mode), "w"
    ) as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerows(c_matrix)

    logger.info("Done.")


if __name__ == "__main__":
    main()
