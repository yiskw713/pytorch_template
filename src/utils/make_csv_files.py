import argparse
import glob
import os
import sys
from typing import Dict, List, Union

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from libs.class_label_map import get_cls2id_map


def get_arguments() -> argparse.Namespace:
    """parse all the arguments from command line inteface return a list of
    parsed arguments."""

    parser = argparse.ArgumentParser(
        description="make csv files for flowers recognition dataset"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="../dataset/flowers/",
        help="path to a dataset dirctory",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./csv",
        help="a directory where csv files will be saved",
    )

    return parser.parse_args()


def split_data(
    data: Dict[str, Dict[str, List[Union[int, str]]]],
    img_paths: List[str],
    cls_name: str,
    cls_id: int,
) -> None:

    for i, path in enumerate(img_paths):
        if i % 5 == 4:
            # for test
            data["test"]["image_path"].append(path)
            data["test"]["label"].append(cls_name)
            data["test"]["class_id"].append(cls_id)
        elif i % 5 == 3:
            # for validation
            data["val"]["image_path"].append(path)
            data["val"]["label"].append(cls_name)
            data["val"]["class_id"].append(cls_id)
        else:
            # for training
            data["train"]["image_path"].append(path)
            data["train"]["label"].append(cls_name)
            data["train"]["class_id"].append(cls_id)


def main() -> None:
    args = get_arguments()

    cls2id_map = get_cls2id_map()

    data: Dict[str, Dict[str, List[Union[int, str]]]] = {
        "train": {
            "image_path": [],
            "class_id": [],
            "label": [],
        },
        "val": {
            "image_path": [],
            "class_id": [],
            "label": [],
        },
        "test": {
            "image_path": [],
            "class_id": [],
            "label": [],
        },
    }

    # 各ディレクトリから画像のパスを指定
    # train : val : test = 6 : 2 : 2 になるように分割
    for cls_name in cls2id_map.keys():
        img_paths = glob.glob(os.path.join(args.dataset_dir, cls_name, "*.jpg"))

        split_data(data, img_paths, cls_name, cls2id_map[cls_name])

    # list を DataFrame に変換
    train_df = pd.DataFrame(
        data["train"],
        columns=["image_path", "class_id", "label"],
    )

    val_df = pd.DataFrame(
        data["val"],
        columns=["image_path", "class_id", "label"],
    )

    test_df = pd.DataFrame(
        data["test"],
        columns=["image_path", "class_id", "label"],
    )

    # 保存ディレクトリがなければ，作成
    os.makedirs(args.save_dir, exist_ok=True)

    # 保存
    train_df.to_csv(os.path.join(args.save_dir, "train.csv"), index=None)
    val_df.to_csv(os.path.join(args.save_dir, "val.csv"), index=None)
    test_df.to_csv(os.path.join(args.save_dir, "test.csv"), index=None)

    print("Finished making csv files.")


if __name__ == "__main__":
    main()
