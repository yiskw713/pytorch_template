import dataclasses
from logging import getLogger

logger = getLogger(__name__)

__all__ = ["DATASET_CSVS"]


@dataclasses.dataclass(frozen=True)
class DatasetCSV:
    train: str
    val: str
    test: str


DATASET_CSVS = {
    # paths from `src` directory
    "flower": DatasetCSV(
        train="./csv/train.csv",
        val="./csv/val.csv",
        test="./csv/test.csv",
    ),
    "dammy": DatasetCSV(
        train="./csv/dammy/train.csv",
        val="./csv/dammy/val.csv",
        test="./csv/dammy/test.csv",
    ),
    # paths to the csv files for pytest is from project root
    "pytest": DatasetCSV(
        train="./tests/sample/pytest_train.csv",
        val="./tests/sample/pytest_val.csv",
        test="./tests/sample/pytest_test.csv",
    ),
}
