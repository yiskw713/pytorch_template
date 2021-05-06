import copy

import numpy as np
import pytest
import torch
import torch.optim as optim
from pytest_mock import MockFixture
from torchvision import transforms

from src.libs.dataset import get_dataloader
from src.libs.helper import do_one_iteration, evaluate, train
from src.libs.loss_fn import get_criterion
from src.libs.models import get_model


@pytest.fixture()
def sample():
    img = torch.randn(2, 3, 112, 112)
    class_id = torch.tensor([0, 1]).long()
    label = ["daisy", "dandelion"]

    return {"img": img, "class_id": class_id, "label": label}


@pytest.fixture()
def model_optimizer():
    model = get_model("resnet18", 5)
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    return (model, optimizer)


@pytest.fixture()
def criterion():
    return get_criterion()


def test_do_one_iteration1(sample, model_optimizer, criterion):
    # check iteration for training
    model, optimizer = model_optimizer
    original_model = copy.deepcopy(model)

    batch_size, loss, acc1, gt, pred = do_one_iteration(
        sample, model, criterion, "cpu", "train", optimizer
    )

    assert batch_size == 2
    assert loss > 0
    assert 0 <= acc1 <= 100.0
    assert np.all(gt == np.array([0, 1]))
    assert pred.shape == (2,)

    # check if models have the same weights
    # https://discuss.pytorch.org/t/check-if-models-have-same-weights/4351
    for key_item1, key_item2 in zip(
        model.state_dict().items(), original_model.state_dict().items()
    ):
        # if the weights are completely identical, training does not work.
        assert not torch.equal(key_item1[1], key_item2[1])


def test_do_one_iteration2(sample, model_optimizer, criterion):
    # check iteration for evaluation
    model, optimizer = model_optimizer
    original_model = copy.deepcopy(model)

    model.eval()
    batch_size, loss, acc1, gt, pred = do_one_iteration(
        sample, model, criterion, "cpu", "evaluate"
    )

    assert batch_size == 2
    assert loss > 0
    assert 0 <= acc1 <= 100.0
    assert np.all(gt == np.array([0, 1]))
    assert pred.shape == (2,)

    # check if models have the same weights
    # https://discuss.pytorch.org/t/check-if-models-have-same-weights/4351
    for key_item1, key_item2 in zip(
        model.state_dict().items(), original_model.state_dict().items()
    ):
        # if the weights are completely identical, training does not work.
        assert torch.equal(key_item1[1], key_item2[1])


def test_do_one_iteration3(sample, model_optimizer, criterion):
    model, optimizer = model_optimizer
    with pytest.raises(ValueError):
        do_one_iteration(sample, model, criterion, "cpu", "test")

    with pytest.raises(ValueError):
        do_one_iteration(sample, model, criterion, "cpu", "train")


def test_train(mocker: MockFixture, model_optimizer, criterion):
    model, optimizer = model_optimizer

    mocker.patch("src.libs.helper.do_one_iteration").return_value = (
        2,
        0.1,
        50.0,
        np.array([0, 1]),
        np.array([1, 1]),
    )

    loader = get_dataloader(
        "pytest",
        "train",
        batch_size=2,
        shuffle=False,
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

    # make small dataset
    loader.dataset.df = loader.dataset.df[:10]

    loss, acc1, f1s = train(
        loader, model, criterion, optimizer, 0, "cpu", interval_of_progress=1
    )

    assert model.training
    assert loss == 0.1
    assert acc1 == 50.0
    assert 0 <= f1s <= 1.0


def test_evaluate(mocker: MockFixture, model_optimizer, criterion):
    model, _ = model_optimizer

    mocker.patch("src.libs.helper.do_one_iteration").return_value = (
        2,
        0.1,
        50.0,
        np.array([0, 1]),
        np.array([1, 1]),
    )

    loader = get_dataloader(
        "pytest",
        "test",
        batch_size=2,
        shuffle=False,
        num_workers=1,
        pin_memory=False,
        drop_last=False,
        transform=transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        ),
    )

    # make small dataset
    loader.dataset.df = loader.dataset.df[:10]
    n_classes = loader.dataset.get_n_classes()

    loss, acc1, f1s, c_matrix = evaluate(loader, model, criterion, "cpu")

    assert not model.training
    assert loss == 0.1
    assert acc1 == 50.0
    assert 0 <= f1s <= 1.0
    assert c_matrix.shape == (n_classes, n_classes)
