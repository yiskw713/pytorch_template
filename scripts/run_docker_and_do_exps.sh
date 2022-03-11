# TODO: add train and create config codes.
PROJECT_ROOT=$(dirname $(cd $(dirname $0) && pwd))

docker container run \
    --gpus all --shm-size=8g \
    -d --rm --restart always \
    -v /etc/group:/etc/group:ro \
    -v /etc/password:/etc/password:ro \
    -u $(id -u $USER):$(id -g $USER) \
    -p 8888:8888 \
    -v $PROJECT_ROOT:/project \
    --name pytorch_template_cnt \
    pytorch_template_image python src/train.py
