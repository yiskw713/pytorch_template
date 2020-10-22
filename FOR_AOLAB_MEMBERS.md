# Image Classification Template for B4 in Aolab

画像分類問題のコードの一例です．研究室での勉強用コードです．

## Requirements

* python >= 3.7
* pytorch >= 1.0
* pyyaml
* scikit-learn
* wandb

必要なpythonパッケージは，`pip install -r requirements.txt` でインストールできます．

## Dataset

Flowers Recognition Dataset を使います．
ダウンロードは[こちら](https://www.kaggle.com/alxmamaev/flowers-recognition/download)から．

## Directory Structure

以下のようなディレクトリ構成を想定しています．
基本的には，`libs`には `train.py` や `test.py` などメインのスクリプトを実行するのに必要なスクリプトをおきます．
`utils`にはそれ以外のスクリプト(例: `make_csv_files.py` や結果を可視化するスクリプトなど)を配置します．
データセットの場所は，以下の通りでなくても大丈夫です．

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

## コードを書く手順

以下の順でコードを書いていきます．

1. データセットクラスのための csv file の作成 (`utils/make_csv_files.py`)
1. configファイルの自動生成 (`utils/make_configs.py`)
1. データセットクラスの作成 (`libs/dataset.py`)
    * データセットの画像に対する前処理のコード (`libs/transformer.py`)
    * 前処理のコードに必要な平均値，標準偏差を書いたスクリプト (`libs/mean.py`)
    * クラスのindex とラベルの対応を記すスクリプト (`libs/class_weight_map.py`)
1. モデルの定義 (e.g. `libs/models/mymodel.py`)
1. ロス関数の定義 (e.g. `libs/loss_fn/myloss.py`)
1. その他学習に必要なコード (`libs/checkpoint.py`, `libs/class_weight.py`, `libs/metric.py`)
1. 学習のコード (`train.py`)
    * config file を用いる (`result/r18_lr0.0005/config.yaml`)
1. 評価するためのコード (`evaluate.py`)
1. 学習とテストのコードをいっぺんに回すためのシェルスクリプトの作成 (`experiment.sh`)

このコードでは，データセットの画像とラベルのペアをあらかじめ csv file に書き出します．
csv に書き出す理由は，ラベル以外に情報を含めるのが簡単だったり，json file などと比べて見やすいと個人的に思うからです．
また，ラベル以外のメタ情報を使いたいときなども，それらの処理が容易だからです．

また学習を回す際は，configuration を書いたファイルを作成して，それを読み込むような設定にしています．
`argparse` などで細かく実験設定を記載するよりも，楽で見やすいし，何より実験設定を保存して置いたり，スクリプトを一気に回すことが容易だからです．

## Experiment

以下のスクリプトを実行することで実験が回る．`utils/make_configs.py`を実行すると自動でconfiguration fileを生成してくれる．
変更したいパラメータをコマンドライン引数として実行してください．

```shell
sh experiment.sh
```

## その他

コードの可読性は本当に大事です．pep8は守ったり，コメントはできる限り残すようにした方がいいと思います．

* black
* flake8
* isort

などを使ってコードを綺麗に整形しましょう．
