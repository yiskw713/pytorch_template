import os
import pandas as pd
import sys
import torch

from PIL import Image
from torch.utils.data import Dataset


class FlowersDataset(Dataset):
    """Flowers Recognition Dataset
    今回のデータセットは，torchvision.datasets.ImageFolder でも実装可能だが，
    csv を作る練習をかねて，データセットクラスを自作している．
    """

    def __init__(self, config, transform=None, mode='training'):
        super().__init__()

        self.config = config

        if mode == 'training':
            csv_path = os.path.join(self.config.csv_dir, 'train.csv')
        elif mode == 'validation':
            csv_path = os.path.join(self.config.csv_dir, 'val.csv')
        elif mode == 'test':
            csv_path = os.path.join(self.config.csv_dir, 'test.csv')
        else:
            print('You have to choose training or validation as the dataset mode')
            sys.exit(1)

        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['image_path']
        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)

        cls_id = self.df.iloc[idx]['cls_id']
        cls_id = torch.tensor(cls_id).long()

        label = self.df.iloc[idx]['label']

        sample = {
            'img': img,
            'cls_id': cls_id,
            'label': label,
            'img_path': img_path
        }

        return sample
