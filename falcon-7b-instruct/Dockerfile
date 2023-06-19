FROM nvidia/cuda:11.7.1-devel-ubuntu22.04
ENV MODEL="/model/saved"
ENV MODEL_ID="tiiuae/falcon-7b-instruct"
ENV MODEL_LOCAL_FILES_ONLY="true"
ENV SERVER_MODEL_NAME="falcon-7b-instruct"
ENV MODEL_LOAD_IN_8BIT="true"
ENV MODEL_TRUST_REMOTE_CODE="true"
ENV PORT=8080
ENV PATH="$PATH:/model/scripts"

RUN mkdir -p /model/saved /model/trained /model/logs /model/scripts /model/src
WORKDIR /model

RUN --mount=type=cache,target=/var/cache/apt --mount=type=cache,target=/var/lib/apt <<EOF
set -ex
apt-get update
apt-get -y --no-install-recommends install python3 python3-pip git
rm -rf /var/lib/apt/lists/*
EOF

RUN ln -s /usr/bin/python3 /usr/bin/python
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache pip install -r /model/requirements.txt
COPY src/build.ipynb /model/src/
COPY scripts/build.sh /model/scripts/
RUN build.sh

COPY src/* ./src/
COPY scripts/* ./scripts/
COPY sample-data sample-data

CMD serve.sh
EXPOSE $PORT
