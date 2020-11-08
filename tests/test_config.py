from typing import Any, Dict

import pytest
from _pytest.capture import CaptureFixture

from src.libs.config import Config


class TestConfig(object):
    @pytest.fixture()
    def base_dict(self) -> Dict[str, Any]:
        _dict = {
            "batch_size": 32,
            "height": 224,
            "learning_rate": 0.0003,
            "max_epoch": 50,
            "model": "resnet18",
            "num_workers": 2,
            "pretrained": True,
            "test_csv": "./csv/test.csv",
            "train_csv": "./csv/train.csv",
            "use_class_weight": True,
            "val_csv": "./csv/val.csv",
            "width": 224,
        }
        return _dict

    def test_type_check(self, base_dict: Dict[str, Any]) -> None:
        for k in base_dict.keys():
            _dict = base_dict.copy()
            _dict[k] = ["dammy", "list"]

            with pytest.raises(TypeError):
                Config(**_dict)

    def test_post_init(self, base_dict: Dict[str, Any], capfd: CaptureFixture) -> None:
        Config(**base_dict)

        # test printed string
        _, err = capfd.readouterr()
        assert err == ""
