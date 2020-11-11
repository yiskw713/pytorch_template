from typing import Any, Dict

import pytest
from _pytest.capture import CaptureFixture

from src.libs.config import Config, convert_list2tuple, get_config


@pytest.fixture()
def base_dict() -> Dict[str, Any]:
    _dict = {
        "batch_size": 32,
        "height": 224,
        "learning_rate": 0.0003,
        "max_epoch": 50,
        "model": "resnet18",
        "num_workers": 2,
        "pretrained": True,
        "test_csv": "./csv/test.csv",
        "topk": (1, 3, 5),
        "train_csv": "./csv/train.csv",
        "use_class_weight": True,
        "val_csv": "./csv/val.csv",
        "width": 224,
    }
    return _dict


class TestConfig(object):
    def test_type_check(self, base_dict: Dict[str, Any]) -> None:
        for k in base_dict.keys():
            _dict = base_dict.copy()
            if k == "pretrained" or k == "use_class_weight":
                _dict[k] = "test"
            else:
                _dict[k] = True

            with pytest.raises(TypeError):
                Config(**_dict)

    def test_type_check_element(self, base_dict: Dict[str, Any]) -> None:
        for val in [("train", "test"), (True, False), (1.2, 2.5)]:
            _dict = base_dict.copy()
            _dict["topk"] = val

            with pytest.raises(TypeError):
                Config(**_dict)

    def test_post_init(self, base_dict: Dict[str, Any], capfd: CaptureFixture) -> None:
        Config(**base_dict)

        # test printed string
        _, err = capfd.readouterr()
        assert err == ""


def test_convert_list2tuple() -> None:
    _dict = {"test": [1, 2, 3], "train": ["hoge", "foo"], "validation": [True, False]}

    _dict = convert_list2tuple(_dict)

    for val in _dict.values():
        assert isinstance(val, tuple)


def test_get_config(base_dict: Dict[str, Any]) -> None:
    config = get_config("tests/sample/config.yaml")

    for key, val in base_dict.items():
        assert val == getattr(config, key)
