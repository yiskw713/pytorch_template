import pytest
from pytest_mock import MockFixture

from src.libs.device import get_device


@pytest.mark.parametrize(
    ("cuda_available", "allow_only_gpu", "expected"),
    [
        (False, False, "cpu"),
        (True, True, "cuda"),
        (True, False, "cuda"),
    ],
)
def test_get_device1(
    mocker: MockFixture, cuda_available: bool, allow_only_gpu: bool, expected: str
) -> None:
    mocker.patch("torch.cuda.is_available").return_value = cuda_available

    assert get_device(allow_only_gpu=allow_only_gpu) == expected


def test_get_device2(mocker: MockFixture):
    mocker.patch("torch.cuda.is_available").return_value = False

    with pytest.raises(ValueError):
        get_device(allow_only_gpu=True)
