from src.libs.mean_std import get_mean, get_std


def test_get_mean() -> None:
    mean = get_mean(norm_value=1.0)
    assert mean == [123.675, 116.28, 103.53]


def test_get_std() -> None:
    std = get_std(norm_value=1.0)
    return std == [58.395, 57.12, 57.375]
