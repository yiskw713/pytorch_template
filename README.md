# Image Classification Template

project for pytorch implementation example of image classification

## Requirements

* python >= 3.7
* pytorch >= 1.0
* pyyaml
* scikit-learn
* wandb

Please run `pip install -r requirements.txt` to install the necessary packages.

## Dataset

Flowers Recognition Dataset
Download the dataset from [HERE](https://www.kaggle.com/alxmamaev/flowers-recognition/download)．

## Directory Structure

```Directory Structure
root/ ──── csv/
        ├─ libs/
        ├─ result/
        ├─ utils/
        ├─ dataset ─── flowers/
        ├─ scripts ─── experiment.sh
        ├ .gitignore
        ├ README.md
        ├ FOR_AOLAB_MEMBERS.md
        ├ requirements.txt
        ├ evaluate.py
        └ train.py
```

## Features

* configuration class using `dataclasses.dataclass` (`libs/config.py`)
  * type check.
  * detection of unnecessary / extra parameters in a specified configuration.
  * `dataclass` is an immutable object, which prevents the setting from being changed by mistake.
* automatically generating configuration files (`utils/make_configs.py`)
  * e.g.) run `python utils/make_configs.py --model resnet18 resnet30 resnet50 --learning_rate 0.001 0.0001`,
  then you can get all of the combinations with `model` and `learning_rate` (total 6 config files)
  while the other parameters are set by default as described in `libs/config.py`.
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

- [ ] test code
- [ ] mypy
- [ ] pre-commit formatting

## License

This repository is released under the [MIT License](./LICENSE)
