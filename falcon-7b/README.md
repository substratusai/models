# Model: tiiuae/falcon-7b

Falcon-7B is a 7B parameters causal decoder-only model built by TII and trained on 1,500B tokens of RefinedWeb enhanced with curated corpora. It is made available under the Apache 2.0 license.


## Usage

### Building

```sh
docker build -t falcon7b .
```

### Running the image and inference server
Run inference server with HTTP endpoint:
```sh
docker run --gpus all --runtime nvidia -d -p 8080:8080 falcon-7b
```
This might take a few minutes to load the model.

Once the model is loaded try running inference by visiting:
[http://localhost:8080](http://localhost:8080).

### Fine tuning the model
```
docker run -v $(pwd)/trained:/trained falcon-7b python train.py ./sample-data/k8s-instructions.jsonl
```
