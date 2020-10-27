import pandas as pd
import torch


def get_class_num(train_csv_file: str) -> torch.Tensor:
    """
    get the number of samples in each class
    Args:
        train_csv_file: the path to the train csv file
    """

    df = pd.read_csv(train_csv_file)
    n_classes = df["class_id"].nunique()

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


def get_class_weight(train_csv_file: str) -> torch.Tensor:
    """Class weight for CrossEntropy in Flowers Recognition Dataset Class
    weight is calculated in the way described in:

    D. Eigen and R. Fergus, “Predicting depth, surface normals and semantic labels
    with a common multi-scale convolutional architecture,” in ICCV 2015,
    openaccess:
    https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Eigen_Predicting_Depth_Surface_ICCV_2015_paper.pdf
    """

    class_num = get_class_num(train_csv_file)
    total = class_num.sum().item()
    frequency = class_num.float() / total
    median = torch.median(frequency)
    class_weight = median / frequency

    return class_weight
