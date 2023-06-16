FROM python:3.10

RUN mkdir /app
WORKDIR /app

COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir /built /trained

COPY . .

RUN python src/build.py
