# Pytorch Template

![status-badge](https://github.com/yiskw713/pytorch_template/workflows/mypy_pytest/badge.svg)

project for pytorch implementation example of image classification

## Requirements

* python >= 3.7
* pytorch >= 1.0
* pyyaml
* scikit-learn
* [wandb](https://wandb.ai/)
* [pre-commit](https://pre-commit.com/) (for pre-commit formatting, type check and testing)
* [hiddenlayer](https://github.com/waleedka/hiddenlayer)
* [graphviz](https://graphviz.gitlab.io/download/)
* [python wrapper for graphviz](https://github.com/xflr6/graphviz)

Please run `poetry install` to install the necessary packages.

You can also setup the environment using docker and docker-compose.

## Dataset

Flowers Recognition Dataset
Download the dataset from [HERE](https://www.kaggle.com/alxmamaev/flowers-recognition/download)．

## Directory Structure

```Directory Structure
.
├── docs/
├── LICENSE
├── README.md
├── dataset/
│   └── flowers/
├── pyproject.toml
├── .gitignore
├── .gitattributes
├── .pre-commit-config.yaml
├── poetry.lock
├── docker-compose.yaml
├── Dockerfile
├── tests/
└── src/
    ├── csv
    ├── libs/
    ├── utils
    ├── notebook/
    ├── result/
    ├── scripts/
    │   └── experiment.sh
    ├── train.py
    └── evaluate.py
```

## Features

* configuration class using `dataclasses.dataclass` (`libs/config.py`)
  * type check.
  * detection of unnecessary / extra parameters in a specified configuration.
  * `dataclass` is an immutable object,
  which prevents the setting from being changed by mistake.
* automatically generating configuration files (`utils/make_configs.py`)
  * e.g.) run this command

  ```bash
  python utils/make_configs.py --model resnet18 resnet30 resnet50 --learning_rate 0.001 0.0001 --dataset_name flower
  ```

  then you can get all of the combinations with `model` and `learning_rate` (total 6 config files),
  while the other parameters are set by default as described in `libs/config.py`.

  You can choose which data you use in experiment by specifying `dataset_name`.
  The lists of data for training, validation and testing are saved as csv files.
  You can see the paths to them in `libs/dataset_csv.py` and get them corresponding to `dataset_name`.
  If you want to use another dataset, please add csv files and the paths in `DATASET_CSVS` in `libs/dataset_csv.py`.

  You can also set tuple object parameters in configs like the below.

  ```bash
  python utils/make_configs.py --model resnet18 --topk 1 3 --topk 1 3 5
  ```

  By running this, you can get two configurations,
  in one of which topk parameter is (1, 3)
  and in the other topk parameter is (1, 3, 5).
* running all the experiments by running shell scripts (`scripts/experiment.sh`)
* support type annotation (`typing`)
* code formatting with `black`, `isort` and `flake8`
* visualize model for debug using [`hiddenlayer`](https://github.com/waleedka/hiddenlayer) (`src/utils/visualize_model.py`)

## Experiment

Please see `scripts/experiment.sh` for the detail.
You can set configurations and run all the experiments by the below command.

```sh
sh scripts/experiment.sh
```

### Setup dependencies

If you use local environment, then run

```sh
poetry install
```

If you use docker, then run

```sh
docker-compose up -d --build
docker-compose run mlserver bash
```

### training

```sh
python train.py ./result/xxxx/config.yaml
```

### evaluation

```shell
python evaluate.py ./result/xxxx/config.yaml validation
python evaluate.py ./result/xxxx/config.yaml test
```

### Model visualization

```shell
python utils/visualize_model.py MODEL_NAME
```

## Formatting

* black
* flake8
* isort

## TODO

* [x] pytorch implementation of image classification
* [x] configuration class using `dataclasses.dataclass`
* [x] auto generation of config yaml files
* [x] shell script to run all the experiment
* [x] support `typing` (type annotation)
* [x] test code (run testing with pre-commit check)
* [x] `mypy` (pre-commit check)
* [x] formatting (pre-commit `isort`, `black` and `flake8`)
* [x] calculate cyclomatic complexity / expression complexity / cognitive complexity (`flake8` extension)
* [x] CI for testing using GitHub Actions
* [x] visualization of models
* [x] add Dockerfile and docker-compose.yaml

## License

This repository is released under the [MIT License](./LICENSE)
