import argparse
import glob
import pandas as pd
import os


def get_arguments():
    '''
    parse all the arguments from command line inteface
    return a list of parsed arguments
    '''

    parser = argparse.ArgumentParser(
        description='train a network for flowers recognition dataset')
    parser.add_argument(
        '--dataset_dir',
        type=str,
        default='./dataset/flowers/',
        help='path to a dataset dirctory'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='./csv',
        help='a directory where csv files will be saved'
    )

    return parser.parse_args()


cls2id_map = {
    'daisy': 0, 'dandelion': 1, 'rose': 2, 'sunflower': 3, 'tulip': 4
}


def main():
    args = get_arguments()

    # img や label を保存するリスト
    train_img_paths = []
    val_img_paths = []
    test_img_paths = []

    train_cls_ids = []
    val_cls_ids = []
    test_cls_ids = []

    train_cls_labels = []
    val_cls_labels = []
    test_cls_labels = []

    # 各ディレクトリから画像のパスを指定
    # train : val : test = 6 : 2 : 2 になるように分割
    for cls in cls2id_map.keys():
        img_paths = glob.glob(os.path.join(args.dataset_dir, cls, '*.jpg'))

        for i, path in enumerate(img_paths):
            if i % 5 == 4:
                # for test
                test_img_paths.append(path)
                test_cls_ids.append(cls2id_map[cls])
                test_cls_labels.append(cls)
            elif i % 5 == 3:
                # for validation
                val_img_paths.append(path)
                val_cls_ids.append(cls2id_map[cls])
                val_cls_labels.append(cls)
            else:
                # for training
                train_img_paths.append(path)
                train_cls_ids.append(cls2id_map[cls])
                train_cls_labels.append(cls)

    # list を DataFrame に変換
    train_df = pd.DataFrame({
        "image_path": train_img_paths,
        "class_id": train_cls_ids,
        "label": train_cls_labels},
        columns=["image_path", "class_id", "label"]
    )

    val_df = pd.DataFrame({
        "image_path": val_img_paths,
        "class_id": val_cls_ids,
        "label": val_cls_labels},
        columns=["image_path", "class_id", "label"]
    )

    test_df = pd.DataFrame({
        "image_path": test_img_paths,
        "class_id": test_cls_ids,
        "label": test_cls_labels},
        columns=["image_path", "class_id", "label"]
    )

    # 保存ディレクトリがなければ，作成
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    # 保存
    train_df.to_csv(os.path.join(args.save_dir, 'train.csv'), index=None)
    val_df.to_csv(os.path.join(args.save_dir, 'val.csv'), index=None)
    test_df.to_csv(os.path.join(args.save_dir, 'test.csv'), index=None)

    print("Done")


if __name__ == "__main__":
    main()
