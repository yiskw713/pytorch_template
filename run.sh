# training
python train.py ./result/cfg1/config.yaml --resume
python train.py ./result/cfg2/config.yaml --resume

# test for cfg1
python test.py ./result/cfg1/config.yaml validation
python test.py ./result/cfg1/config.yaml test

# test for cfg2
python test.py ./result/cfg2/config.yaml validation
python test.py ./result/cfg2/config.yaml test
