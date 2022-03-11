PROJECT_ROOT=$(dirname $(cd $(dirname $0) && pwd))

docker image build -t pytorch_template_image \
    -f $PROJECT_ROOT/dockerfiles/Dockerfile \
    $PROJECT_ROOT
