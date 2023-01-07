import datetime
import os
import sys
import time
from typing import Optional

import click
import torch
import torch.optim as optim
import wandb
from loguru import logger

from libs.class_id_map import get_cls2id_map
from libs.config import get_config
from libs.dataset import get_dataloader
from libs.helper import evaluate, train
from libs.loss_fn import get_criterion
from libs.models import get_model
from libs.transform import get_test_transform, get_train_transform
from libs.utils import (
    get_device,
    resume,
    save_checkpoint,
    set_seed,
    setup_logger,
    MetricLogger,
)


@click.command()
@click.option("config_path", type=str, help="path to a config file.")
@click.option(
    "--checkpoint_path",
    type=str,
    default=None,
    help="Path to a checkpoint which you want to resume training from",
)
@click.option(
    "--info",
    "log_level",
    flag_value="INFO",
    default=True,
    help="Add --info option if you want to see INFO-level logs.",
)
@click.option(
    "--debug",
    "log_level",
    flag_value="DEBUG",
    help="Add --debug option if you want to see DEBUG-level logs.",
)
@click.option(
    "--use_wandb", is_flag=True, help="Add --use_wandb option if you want to use wandb."
)
@click.option("--supress_stdout", help="Supress stdout or not.", is_flag=True)
@click.option("--seed", type=int, default=0, help="random seed")
@click.option("--num_workers", type=int, default=4, help="num_workers for dataloader")
@click.option("--topk", type=int, multiple=True, default=(1, 5), help="topk accuracy")
@logger.catch(onerror=lambda _: sys.exit(1))
def main(
    config_path: str,
    checkpoint_path: Optional[str],
    log_level: str,
    use_wandb: bool,
    supress_stdout: bool,
    seed: bool,
    num_workers: int,
) -> None:
    """Traning a model based on a yaml configuration."""

    # save log files in the directory which contains config file.
    result_path = os.path.dirname(config_path)
    experiment_name = os.path.basename(result_path)

    # setting logger configuration
    log_path = os.path.join(
        result_path, f"{datetime.datetime.now():%Y-%m-%d}_train.log"
    )
    setup_logger(
        log_path,
        log_level=log_level,
        supress_stdout=supress_stdout,
    )

    # fix seed
    set_seed()

    # configuration
    config = get_config(config_path)

    # cpu or cuda
    device = get_device(allow_only_gpu=False)

    # Dataloader
    train_transform = get_train_transform(config.height, config.width)
    val_transform = get_test_transform()

    train_loader = get_dataloader(
        config.dataset_name,
        "train",
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        transform=train_transform,
        seed=seed,
    )

    val_loader = get_dataloader(
        config.dataset_name,
        "val",
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        transform=val_transform,
        seed=seed,
    )

    # the number of classes
    n_classes = len(get_cls2id_map())

    # define a model
    model = get_model(config.model, n_classes, pretrained=config.pretrained)

    # send the model to cuda/cpu
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # keep training and validation log
    begin_epoch = 0
    best_loss = float("inf")

    # resume if you want
    if checkpoint_path is not None:
        begin_epoch, model, optimizer, best_loss = resume(
            checkpoint_path, model, optimizer
        )

    log_path = os.path.join(result_path, "log.csv")
    metric_logger = MetricLogger(log_path, resume=checkpoint_path is not None)

    # criterion for loss
    criterion = get_criterion(config.use_class_weight, config.dataset_name, device)

    # Weights and biases
    if use_wandb:
        wandb.init(
            name=experiment_name,
            config=config,
            project="image_classification_template",
            job_type="training",
            dir="./wandb_result/",
        )
        # Magic
        wandb.watch(model, log="all")

    # train and validate model
    logger.info("Start training.")

    for epoch in range(begin_epoch, config.max_epoch):
        # training
        start = time.time()
        train_result = train(train_loader, model, criterion, optimizer, epoch, device)
        train_time = int(time.time() - start) // 60
        train_result["train time [min]"] = train_time

        # validation
        start = time.time()
        val_result, c_matrix = evaluate(val_loader, model, criterion, device)
        val_time = int(time.time() - start) // 60
        val_result["val time [min]"] = val_time

        # save a model if top1 acc is higher than ever
        if best_loss > val_result["Evaluation Loss"]:
            best_loss = val_result["Evaluation Loss"]
            torch.save(
                model.state_dict(),
                os.path.join(result_path, "best_model.prm"),
            )

        # save checkpoint every epoch
        save_checkpoint(
            os.path.join(result_path, "checkpoint.pth"),
            epoch,
            model,
            optimizer,
            best_loss,
        )

        # write logs to dataframe and csv file
        metric_logger.update(
            epoch=epoch,
            lr=optimizer.param_groups[0]["lr"],
            **train_result,
            **val_result,
        )

        # save logs to wandb
        if use_wandb:
            wandb.log(
                {
                    "lr": optimizer.param_groups[0]["lr"],
                    **train_result,
                    **val_result,
                },
                step=epoch,
            )

    # delete checkpoint
    os.remove(os.path.join(result_path, "checkpoint.pth"))

    logger.info("Done")


if __name__ == "__main__":
    main()
