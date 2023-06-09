FROM nvidia/cuda:11.7.1-devel-ubuntu22.04
ENV MODEL="/built/"
ENV MODEL_LOCAL_FILES_ONLY="true"
ENV SERVER_MODEL_NAME="falcon-7b-instruct"
ENV MODEL_LOAD_IN_8BIT="true"
ENV MODEL_TRUST_REMOTE_CODE="true"
ENV PORT=8080

RUN mkdir /app /built /trained
WORKDIR /app

RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update \
    && apt-get -y --no-install-recommends install \
    python3 python3-dev python3-pip git git-lfs

RUN git lfs install
RUN --mount=type=cache,target=/tmp/$SERVER_MODEL_NAME DEPTH=1 rm -rf /tmp/$SERVER_MODEL_NAME/.git/index.lock; \
    git clone https://huggingface.co/tiiuae/$SERVER_MODEL_NAME /tmp/$SERVER_MODEL_NAME; \
    cd /tmp/$SERVER_MODEL_NAME && git reset --hard HEAD && git pull && cp -r /tmp/$SERVER_MODEL_NAME/* /built/

RUN ln -s /usr/bin/python3 /usr/bin/python

COPY requirements.txt /app

RUN --mount=type=cache,target=/root/.cache pip install --no-cache-dir -r requirements.txt


COPY *.py .
COPY serve.sh .
COPY sample-data sample-data

CMD ./serve.sh
EXPOSE $PORT
