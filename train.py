import argparse
import os
import sys
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torchvision.transforms import (ColorJitter, Compose, Normalize,
                                    RandomHorizontalFlip, RandomResizedCrop,
                                    ToTensor)

import wandb
from libs.checkpoint import resume, save_checkpoint
from libs.class_label_map import get_cls2id_map
from libs.class_weight import get_class_weight
from libs.config import Config
from libs.dataset import get_dataloader
from libs.helper import train, validate
from libs.mean_std import get_mean, get_std
from libs.models import get_model


def get_arguments() -> argparse.Namespace:
    """
    parse all the arguments from command line inteface
    return a list of parsed arguments
    """

    parser = argparse.ArgumentParser(
        description="""
        train a network for image classification with Flowers Recognition Dataset.
        """
    )
    parser.add_argument("config", type=str, help="path of a config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Add --resume option if you start training from checkpoint.",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Add --no_wandb option if you do not want to use wandb.",
    )

    return parser.parse_args()


def main() -> None:
    args = get_arguments()

    # configuration
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
    CONFIG = Config(**config_dict)

    # save log files in the directory which contains config file.
    result_path = os.path.dirname(args.config)

    # Weights and biases
    if not args.no_wandb:
        wandb.init(
            config=CONFIG,
            project="image_classification_template",
            job_type="training",
        )

    # cpu or cuda
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("You have to use GPUs because training CNN is computationally expensive.")
        sys.exit(1)
    else:
        torch.backends.cudnn.benchmark = True

    # Dataloader
    train_transform = Compose(
        [
            RandomResizedCrop(size=(CONFIG.height, CONFIG.width)),
            RandomHorizontalFlip(),
            ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            ToTensor(),
            Normalize(mean=get_mean(), std=get_std()),
        ]
    )

    val_transform = Compose([ToTensor(), Normalize(mean=get_mean(), std=get_std())])

    train_loader = get_dataloader(
        CONFIG.train_csv,
        batch_size=CONFIG.batch_size,
        shuffle=True,
        num_workers=CONFIG.num_workers,
        pin_memory=True,
        drop_last=True,
        transform=train_transform,
    )

    val_loader = get_dataloader(
        CONFIG.val_csv,
        batch_size=1,
        shuffle=False,
        num_workers=CONFIG.num_workers,
        pin_memory=True,
        transform=val_transform,
    )

    # the number of classes
    n_classes = len(get_cls2id_map())

    # define a model
    model = get_model(CONFIG.model, n_classes, pretrained=CONFIG.pretrained)

    # send the model to cuda/cpu
    model.to(device)

    if not args.no_wandb:
        # Magic
        wandb.watch(model, log="all")

    optimizer = optim.Adam(model.parameters(), lr=CONFIG.learning_rate)

    # keep training and validation log
    begin_epoch = 0
    best_loss = float("inf")
    log = pd.DataFrame(
        columns=[
            "epoch",
            "lr",
            "train_time[sec]",
            "train_loss",
            "train_acc@1",
            "train_f1s",
            "val_time[sec]",
            "val_loss",
            "val_acc@1",
            "val_f1s",
        ]
    )

    # resume if you want
    if args.resume:
        resume_path = os.path.join(result_path, "checkpoint.pth")
        begin_epoch, model, optimizer, best_loss = resume(resume_path, model, optimizer)

        log_path = os.path.join(result_path, "log.csv")
        assert os.path.exists(log_path), "there is no checkpoint at the result folder"
        log = pd.read_csv(log_path)

    # criterion for loss
    if CONFIG.class_weight:
        criterion = nn.CrossEntropyLoss(
            weight=get_class_weight(CONFIG.train_csv, n_classes=n_classes).to(device)
        )
    else:
        criterion = nn.CrossEntropyLoss()

    # train and validate model
    print("\n------------------------Start training------------------------\n")

    for epoch in range(begin_epoch, CONFIG.max_epoch):
        # training
        start = time.time()
        train_loss, train_acc1, train_f1s = train(
            train_loader, model, criterion, optimizer, epoch, device
        )
        train_time = int(time.time() - start)

        # validation
        start = time.time()
        val_loss, val_acc1, val_f1s = validate(val_loader, model, criterion, device)
        val_time = int(time.time() - start)

        # save a model if top1 acc is higher than ever
        if best_loss > val_loss:
            best_loss = val_loss
            torch.save(
                model.state_dict(),
                os.path.join(result_path, "best_model.prm"),
            )

        # save checkpoint every epoch
        save_checkpoint(result_path, epoch, model, optimizer, best_loss)

        # write logs to dataframe and csv file
        tmp = pd.Series(
            [
                epoch,
                optimizer.param_groups[0]["lr"],
                train_time,
                train_loss,
                train_acc1,
                train_f1s,
                val_time,
                val_loss,
                val_acc1,
                val_f1s,
            ],
            index=log.columns,
        )

        log = log.append(tmp, ignore_index=True)
        log.to_csv(os.path.join(result_path, "log.csv"), index=False)

        # save logs to wandb
        if not args.no_wandb:
            wandb.log(
                {
                    "lr": optimizer.param_groups[0]["lr"],
                    "train_time[sec]": train_time,
                    "train_loss": train_loss,
                    "train_acc@1": train_acc1,
                    "train_f1s": train_f1s,
                    "val_time[sec]": val_time,
                    "val_loss": val_loss,
                    "val_acc@1": val_acc1,
                    "val_f1s": val_f1s,
                },
                step=epoch,
            )

        print(
            "epoch: {}\tepoch time[sec]: {}\tlr: {}\ttrain loss: {:.4f}\tval loss: {:.4f} val_acc1: {:.5f}\tval_f1s: {:.5f}".format(
                epoch,
                train_time + val_time,
                optimizer.param_groups[0]["lr"],
                train_loss,
                val_loss,
                val_acc1,
                val_f1s,
            )
        )

    # save models
    torch.save(model.state_dict(), os.path.join(result_path, "final_model.prm"))

    print("Done")


if __name__ == "__main__":
    main()
