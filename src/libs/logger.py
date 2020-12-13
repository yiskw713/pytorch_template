import os

import pandas as pd


class TrainLogger(object):
    def __init__(self, log_path: str, resume: bool) -> None:
        self.log_path = log_path
        self.columns = [
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

        if resume:
            if os.path.exists(log_path):
                self.df = pd.DataFrame(columns=self.columns)
            else:
                raise FileNotFoundError("Log file not found.")
        else:
            self.df = pd.DataFrame(columns=self.columns)

    def _save_log(self) -> None:
        self.df.to_csv(self.log_path, index=False)

    def update(self):
        pass
