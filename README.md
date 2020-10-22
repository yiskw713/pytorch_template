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
Download the dataset from [HERE](https://www.kaggle.com/alxmamaev/flowers-recognition/download)から．

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
* automatically generating configuration files (`utils/make_configs.py`)
* running all the experiments by running shell scripts (`scripts/experiment.sh`)

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
