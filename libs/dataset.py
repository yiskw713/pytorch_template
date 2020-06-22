import os
import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Any, Dict, Optional


class FlowersDataset(Dataset):
    """Flowers Recognition Dataset
    今回のデータセットは，torchvision.datasets.ImageFolder でも実装可能だが，
    csv を作る練習をかねて，データセットクラスを自作している．
    """

    def __init__(
        self, csv_file: str, transform: Optional[transforms.Compose] = None
    ) -> None:
        super().__init__()
        assert os.path.exists(csv_file)

        csv_path = os.path.join(csv_file)

        self.df = pd.read_csv(csv_path)
        self.transform = transform

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
