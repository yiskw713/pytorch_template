python utils/build_dataset.py

# training
python train.py ./result/r18_lr0.0005/config.yaml
python train.py ./result/r34_lr0.0005/config.yaml

# test for r18_lr0.0005
python eval.py ./result/r18_lr0.0005/config.yaml validation
python eval.py ./result/r18_lr0.0005/config.yaml test

# test for r34_lr0.0005
python eval.py ./result/r34_lr0.0005/config.yaml validation
python eval.py ./result/r34_lr0.0005/config.yaml test
