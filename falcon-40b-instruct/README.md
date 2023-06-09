# Model: tiiuae/falcon-40b-instruct

Falcon-40B-Instruct is a 40B parameters causal decoder-only model built by TII based on Falcon-40B and finetuned on a mixture of Baize. It is made available under the Apache 2.0 license.

The model requires ~88GB of GPU memory while in 16 bit mode to be able to run inference at a decent speed.
The model is able to run on 4 x L4 GPUs in 16 bit mode. Running the model in 4 bit or 8 bit
mode significantly decreases the inference performance. For example, on 4 x L4 GPUs in 16 bit mode the model is able
to generate ~2.5 tokens / sec. You could also run it on 2 x GPU with 48 GB of memory.

## Usage

### Building

```sh
docker build -t falcon40b-instruct .
```

### Running the image and inference server
Run inference server with HTTP endpoint:
```sh
docker run --gpus all --runtime nvidia -d -p 8080:8080 falcon-40b-instruct
```
This might take a few minutes to load the model.

Once the model is loaded try running inference by submitting a HTTP post
request:
```sh
curl -X POST http://localhost:8080/generate \
   -H "Content-Type: application/json" \
   -d '{"prompt": "Who was the first president of the United States?", "max_new_tokens": 10}'
```

You can also visit the API docs by going to [http://localhost:8080/docs](http://localhost:8080/docs).

### Fine tuning the model
Future roadmap not working yet:
```
docker run -v $(pwd)/trained:/trained falcon-40b-instruct python train.py ./sample-data/k8s-
instructions.jsonl
```
