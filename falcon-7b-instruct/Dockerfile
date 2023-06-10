FROM nvidia/cuda:11.7.1-devel-ubuntu22.04
ENV MODEL="/built/"
ENV MODEL_LOCAL_FILES_ONLY="true"
ENV SERVER_MODEL_NAME="falcon-7b-instruct"
ENV MODEL_LOAD_IN_8BIT="true"
ENV MODEL_TRUST_REMOTE_CODE="true"
ENV PORT=8080

RUN mkdir /app /built /trained
WORKDIR /app

RUN apt-get update \
    && apt-get -y --no-install-recommends install \
    python3 python3-pip git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python
COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt
COPY build.py .
RUN python3 build.py

COPY *.py .
COPY serve.sh .
COPY sample-data sample-data

CMD ./serve.sh
EXPOSE $PORT
