python utils/build_dataset.py

# training
python train.py ./result/cfg1/config.yaml --resume
python train.py ./result/cfg2/config.yaml --resume

# test for cfg1
python eval.py ./result/cfg1/config.yaml validation
python eval.py ./result/cfg1/config.yaml test

# test for cfg2
python eval.py ./result/cfg2/config.yaml validation
python eval.py ./result/cfg2/config.yaml test
