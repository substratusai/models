# Model: tiiuae/falcon-7b-instruct

Falcon-7B-Instruct is a 7B parameters causal decoder-only model built by TII based on Falcon-7B and finetuned on a mixture
of chat/instruct datasets. It is made available under the Apache 2.0 license.

The model by default requires only 14GB of GPU memory for serving. So it can fit on a T4 GPU.
This is because the model is configured to float16 instead of float32.

In future will make bit quantization optional so it can run on even smaller GPUs.

## Usage

### Building

```sh
docker build -t falcon7b-instruct .
```

### Running the image and inference
Run inference server with HTTP endpoint:
```sh
docker run --gpus all --runtime nvidia -d -p 8080:8080 falcon-7b-instruct
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

## TODO
- Fine tuning
- publishing a public image
