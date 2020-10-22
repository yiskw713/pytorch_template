import pprint
import dataclasses


@dataclasses.dataclass
class Config:
    model: str
    pretrained: bool = True

    # whether you use class weight to calculate cross entropy or not
    class_weight: bool = True

    batch_size: int = 32

    width: int = 224
    height: int = 224

    num_workers: int = -1
    max_epoch: int = 50

    learning_rate: float = 0.003

    train_csv: str = "./csv/train.csv"
    val_csv: str = "./csv/val.csv"
    test_csv: str = "./csv/test.csv"

    def __post_init__(self):
        print("-" * 10, "Experiment Configuration", "-" * 10)
        pprint.pprint(dataclasses.asdict(self), width=1)
