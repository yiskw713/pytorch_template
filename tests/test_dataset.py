import pytest
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.libs.dataset import FlowersDataset, get_dataloader


@pytest.mark.parametrize("batch_size", [1, 2])
def test_get_dataloader(batch_size):
    loader = get_dataloader(
        csv_file="./tests/sample/pytest_train.csv",
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=False,
        drop_last=True,
        transform=transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        ),
    )

    assert len(loader) == 8 // batch_size
    assert isinstance(loader, DataLoader)

    for sample in loader:
        assert sample["img"].shape == (batch_size, 3, 224, 224)
        assert sample["img"].dtype == torch.float

        assert isinstance(sample["label"], list)
        assert isinstance(sample["label"][0], str)

        assert sample["class_id"].shape == (batch_size,)
        assert sample["class_id"].dtype == torch.int64
        break


class TestFlowersDataset(object):
    @pytest.fixture()
    def data(self):
        data = FlowersDataset("./tests/sample/pytest_train.csv")
        return data

    def test_len(self, data):
        assert len(data) == 8

    def test_get_n_classes(self, data):
        assert data.get_n_classes() == 1

    def test_getitem(self, data):
        sample = data.__getitem__(0)

        assert "class_id" in sample
        assert "label" in sample
        assert "img" in sample
