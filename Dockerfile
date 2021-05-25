FROM python:3.8.10-buster

WORKDIR /project

# pythonのブロックバッファリングを停止
ENV PYTHONUNBUFFERED=1

RUN pip install poetry

COPY poetry.lock pyproject.toml ./

RUN poetry install

CMD ["/bin/bash"]
