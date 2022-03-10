python utils/make_csv_files.py
python utils/make_configs.py --model resnet18 resnet34 resnet50 --learning_rate 0.003 0.0003

files="./result/*"
for filepath in $files; do
    if [ -d $filepath ] ; then
        python train.py "${filepath}/config.yaml"
        python evaluate.py "${filepath}/config.yaml" validation
        python evaluate.py "${filepath}/config.yaml" test
    fi
done
