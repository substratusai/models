FROM python:3.10

RUN mkdir /app /model /trained-model
WORKDIR /app

COPY requirements.txt /app

RUN pip install --no-cache-dir -r requirements.txt

COPY init.py .
RUN python init.py

COPY sample-data/ sample-data/

COPY *.py .
