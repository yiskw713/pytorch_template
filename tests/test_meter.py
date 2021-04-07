from logging import DEBUG, INFO

import pytest
from _pytest.logging import LogCaptureFixture

from src.libs.meter import AverageMeter, ProgressMeter


@pytest.fixture()
def average_meter() -> AverageMeter:
    meter = AverageMeter("acc", ":.1f")
    meter.update(8.0, 1.0)
    meter.update(12.0, 1.0)
    return meter


def test_average_meter(average_meter: AverageMeter) -> None:
    assert average_meter.get_average() == 10.0


def test_progress_meter(average_meter: AverageMeter, caplog: LogCaptureFixture) -> None:
    caplog.set_level(DEBUG)

    meter = ProgressMeter(2, [average_meter])
    meter.display(2)

    # test logs
    assert (
        "src.libs.meter",
        DEBUG,
        "Progress meter is set up.",
    ) in caplog.record_tuples
    assert (
        "src.libs.meter",
        INFO,
        "[2/2]\tacc 12.0 (avg. 10.0)",
    ) in caplog.record_tuples
