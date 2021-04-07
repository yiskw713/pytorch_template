from logging import getLogger
from typing import Any, Dict, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

__all__ = ["get_dataloader"]

logger = getLogger(__name__)


def get_dataloader(
    csv_file: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool = False,
    transform: Optional[transforms.Compose] = None,
) -> DataLoader:

    data = FlowersDataset(
        csv_file,
        transform=transform,
    )

    dataloader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    return dataloader


class FlowersDataset(Dataset):
    def __init__(
        self, csv_file: str, transform: Optional[transforms.Compose] = None
    ) -> None:
        super().__init__()

        try:
            self.df = pd.read_csv(csv_file)
        except FileNotFoundError("csv file not found.") as e:
            logger.exception(f"{e}")

        self.n_classes = self.df["class_id"].nunique()
        self.transform = transform

        logger.info(f"the number of classes: {self.n_classes}")
        logger.info(f"the number of samples: {len(self.df)}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.df.iloc[idx]["image_path"]
        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)

        cls_id = self.df.iloc[idx]["class_id"]
        cls_id = torch.tensor(cls_id).long()

        label = self.df.iloc[idx]["label"]

        sample = {"img": img, "class_id": cls_id, "label": label, "img_path": img_path}

        return sample

    def get_n_classes(self) -> int:
        return self.n_classes
