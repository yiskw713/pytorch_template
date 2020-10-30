import dataclasses
import pprint


@dataclasses.dataclass
class Config:
    model: str
    pretrained: bool = True

    # whether you use class weight to calculate cross entropy or not
    use_class_weight: bool = True

    batch_size: int = 32

    width: int = 224
    height: int = 224

    num_workers: int = 2
    max_epoch: int = 50

    learning_rate: float = 0.003

    train_csv: str = "./csv/train.csv"
    val_csv: str = "./csv/val.csv"
    test_csv: str = "./csv/test.csv"

    def __post_init__(self) -> None:
        self._type_check()

        print("-" * 10, "Experiment Configuration", "-" * 10)
        pprint.pprint(dataclasses.asdict(self), width=1)

    def _type_check(self) -> None:
        """Reference:
        https://qiita.com/obithree/items/1c2b43ca94e4fbc3aa8d
        """

        _dict = dataclasses.asdict(self)

        for field, field_type in self.__annotations__.items():
            # if you use type annotation class provided by `typing`,
            # you should convert it to the type class used in python.
            # e.g.) List[int] -> list
            if not isinstance(_dict[field], field_type):
                raise TypeError(
                    f"The type of '{field}' field is supposed to be {field_type}."
                )
