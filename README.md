# Image Classification Template for B4 in Aolab
画像分類問題のコードの一例です．研究室での勉強用コードです．

## Requirements
* python 3.x
* pytorch >= 1.0

必要なpythonパッケージは，`pip install -r requirements.txt` でインストールできます．

## Dataset
Flowers Recognition Dataset を使います．
ダウンロードは[こちら](https://www.kaggle.com/alxmamaev/flowers-recognition/download)から．

## Directory Structure
以下のようなディレクトリ構成を想定しています．
基本的には，`libs`には `train.py` や `test.py` などメインのスクリプトを実行するのに必要なスクリプトをおきます．
`utils`にはそれ以外のスクリプト(例: `build_dataset.py` や結果を可視化するスクリプトなど)を配置します．
データセットの場所は，以下の通りでなくても大丈夫です．
```
root/ ──── csv/
        ├─ libs/
        ├─ result/
        ├─ utils/
        ├─ dataset ─── flowers/
        ├ .gitignore
        ├ README.md
        ├ requirements.txt
        ├ eval.py
        └ train.py
```

## コードを書く手順
以下の順でコードを書いていきます．

1. データセットクラスのための csv file の作成 (`utils/build_dataset.py`)
2. データセットクラスの作成 (`libs/dataset.py`)
    - データセットの画像に対する前処理のコード (`libs/transformer.py`)
    - 前処理のコードに必要な平均値，標準偏差を書いたスクリプト (`libs/mean.py`)
    - クラスのindex とラベルの対応を記すスクリプト (`libs/class_weight_map.py`)
3. モデルの定義 (`libs/models/mymodel.py`)
4. ロス関数の定義 (`libs/loss_fn/myloss.py`)
5. その他学習に必要なコード (`libs/checkpoint.py`, `libs/class_weight.py`, `libs/metric.py`)
6. 学習のコード (`train.py`)
    - config file を用いる (`result/cfg1/config.yaml`)
7. 評価するためのコード (`eval.py`)
8. 学習とテストのコードをいっぺんに回すためのシェルスクリプトの作成 (`run.sh`)

このコードでは，データセットの画像とラベルのペアをあらかじめ csv file に書き出します．
csv に書き出す理由は，ラベル以外に情報を含めるのが簡単だったり，json file などと比べて見やすいと個人的に思うからです．
また，ラベル以外のメタ情報を使いたいときなども，それらの処理が容易だからです．

また学習を回す際は，configuration を書いたファイルを作成して，それを読み込むような設定にしています．
`argparse` などで細かく実験設定を記載するよりも，楽でみやすいし，何より実験設定を保存して置いたり，スクリプトを一気に回すことが容易だからです．

## Training
コードを各手順は以下の通り．

``` python utils/build_dataset.py ```

``` python train.py ./result/cfg1/config.yaml --resume ```

``` python eval.py ./result/cfg1/config.yaml validation ```

``` python eval.py ./result/cfg1/config.yaml test ```

ここらへんの処理すべてshell scriptに書き込んで回すのがオススメ．
``` sh run.sh ```

## その他
コードの可読性は本当に大事です．pep8は守ったり，コメントはできる限り残すようにした方がいいと思います．
VSCode を使っている人は[こちら](https://qiita.com/psychoroid/items/2c2acc06c900d2c0c8cb)を参考にコードの自動整形ができるようにしましょう．