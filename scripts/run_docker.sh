PROJECT_ROOT=$(dirname $(cd $(dirname $0) && pwd))

docker container run \
    --gpu all \
    --shm-size=8g \
    -itd --rm \
    -p 8888:8888 \
    -v $PROJECT_ROOT:/project \
    --name pytorch_template_cnt \
    pytorch_template_image bash
