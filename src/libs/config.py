import dataclasses
import os
import pprint
from typing import Any, Dict, Tuple

import yaml

__all__ = ["get_config"]


@dataclasses.dataclass(frozen=True)
class Config:
    """Experimental configuration class."""

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

    topk: Tuple[int, ...] = (1, 3)

    def __post_init__(self) -> None:
        self._type_check()
        self._value_check()

        print("-" * 10, "Experiment Configuration", "-" * 10)
        pprint.pprint(dataclasses.asdict(self), width=1)

    def _value_check(self) -> None:
        if not os.path.exists(self.train_csv):
            raise FileNotFoundError("train_csv is not found")

        if not os.path.exists(self.val_csv):
            raise FileNotFoundError("val_csv is not found")

        if not os.path.exists(self.test_csv):
            raise FileNotFoundError("test_csv is not found")

        if self.max_epoch <= 0:
            raise ValueError("max_epoch must be positive.")

    def _type_check(self) -> None:
        """Reference:
        https://qiita.com/obithree/items/1c2b43ca94e4fbc3aa8d
        """

        _dict = dataclasses.asdict(self)

        for field, field_type in self.__annotations__.items():
            # if you use type annotation class provided by `typing`,
            # you should convert it to the type class used in python.
            # e.g.) Tuple[int] -> tuple
            # https://stackoverflow.com/questions/51171908/extracting-data-from-typing-types

            # check the instance is Tuple or not.
            # https://github.com/zalando/connexion/issues/739
            if hasattr(field_type, "__origin__"):
                # e.g.) Tuple[int].__args__[0] -> `int`
                element_type = field_type.__args__[0]

                # e.g.) Tuple[int].__origin__ -> `tuple`
                field_type = field_type.__origin__

                self._type_check_element(field, _dict[field], element_type)

            # bool is the subclass of int,
            # so need to use `type() is` instead of `isinstance`
            if type(_dict[field]) is not field_type:
                raise TypeError(
                    f"The type of '{field}' field is supposed to be {field_type}."
                )

    def _type_check_element(
        self, field: str, vals: Tuple[Any], element_type: type
    ) -> None:
        for val in vals:
            if type(val) is not element_type:
                raise TypeError(
                    f"The element of '{field}' field is supposed to be {element_type}."
                )


def convert_list2tuple(_dict: Dict[str, Any]) -> Dict[str, Any]:
    # cannot use list in dataclass because mutable defaults are not allowed.
    for key, val in _dict.items():
        if isinstance(val, list):
            _dict[key] = tuple(val)

    return _dict


def get_config(config_path: str) -> Config:
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    config_dict = convert_list2tuple(config_dict)
    config = Config(**config_dict)
    return config
