FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

RUN mkdir /app /built /trained
WORKDIR /app

RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update \
    && apt-get -y --no-install-recommends install \
    python3 python3-dev python3-pip git git-lfs

RUN git lfs install
RUN --mount=type=cache,target=/tmp/falcon-7b-instruct DEPTH=1 git clone https://huggingface.co/tiiuae/falcon-7b-instruct /tmp/falcon-7b-instruct; \
    cd /tmp/falcon-7b-instruct && git pull && cp -r /tmp/falcon-7b-instruct/* /built/

RUN ln -s /usr/bin/python3 /usr/bin/python

COPY requirements.txt /app

RUN --mount=type=cache,target=/root/.cache pip install --no-cache-dir -r requirements.txt


COPY *.py .
COPY sample-data sample-data

CMD python serve.py
EXPOSE 8080
