# Image Classification Template

![status-badge](https://github.com/yiskw713/pytorch_template/workflows/mypy_pytest/badge.svg)

project for pytorch implementation example of image classification

## Requirements

* python >= 3.7
* pytorch >= 1.0
* pyyaml
* scikit-learn
* wandb
* pre-commit (for pre-commit formatting, type check and testing)

Please run `pip install -r requirements.txt` to install the necessary packages.

## Dataset

Flowers Recognition Dataset
Download the dataset from [HERE](https://www.kaggle.com/alxmamaev/flowers-recognition/download)．

## Directory Structure

```Directory Structure
.
├── FOR_AOLAB_MEMBERS.md
├── LICENSE
├── README.md
├── dataset/
│   └── flowers/
├── requirements.txt
├── .gitignore
├── .pre-commit-config
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
  python utils/make_configs.py --model resnet18 resnet30 resnet50 --learning_rate 0.001 0.0001
  ```

  then you can get all of the combinations
  with `model` and `learning_rate` (total 6 config files)
  while the other parameters are set by default
  as described in `libs/config.py`.
* running all the experiments by running shell scripts (`scripts/experiment.sh`)
* support type annotation (`typing`)
* code formatting with `black`, `isort` and `flake8`

## Experiment

Please see `scripts/experiment.sh` for the detail.
You can set configurations and run all the experiments by the below command.

```shell
sh scripts/experiment.sh
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
* [ ] CI for testing using GitHub Actions

## License

This repository is released under the [MIT License](./LICENSE)
