import pandas as pd
import torch

from .class_label_map import get_label2id_map


def get_class_num(train_csv_file='./csv/train.csv', n_classes=len(get_label2id_map())):
    """ get the number of samples in each class """

    df = pd.read_csv(train_csv_file)
    nums = {}
    for i in range(n_classes):
        nums[i] = 0
    for i in range(len(df)):
        nums[df.iloc[i, 1]] += 1
    class_num = []
    for val in nums.values():
        class_num.append(val)
    class_num = torch.tensor(class_num)

    return class_num


def get_class_weight(train_csv_file='./csv/train.csv', n_classes=len(get_label2id_map())):
    """
    Class weight for CrossEntropy in Flowers Recognition Dataset
    Class weight is calculated in the way described in:
        D. Eigen and R. Fergus, “Predicting depth, surface normals and semantic labels with a common multi-scale convolutional architecture,” in ICCV,
        openaccess: https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Eigen_Predicting_Depth_Surface_ICCV_2015_paper.pdf
    """

    class_num = get_class_num(train_csv_file, n_classes)
    total = class_num.sum().item()
    frequency = class_num.float() / total
    median = torch.median(frequency)
    class_weight = median / frequency

    return class_weight
