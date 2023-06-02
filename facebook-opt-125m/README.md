# Model: Facebook Opt-125m

A teeny tiny LLM model (https://huggingface.co/facebook/opt-125m).

## Usage

Build.

```sh
docker build -t facebook-opt-125m .
```

Run inference.

```sh
docker run facebook-opt-125m python infer.py "My favorite color is"
```

Explore wiht a Notebook.

```sh
docker run -p 8888:8888 facebook-opt-125m jupyter notebook --allow-root --ip=0.0.0.0 --NotebookApp.token='' --notebook-dir='/app'
open http://localhost:8888
```

Run Server.

```sh
docker run -p 8080:8080 facebook-opt-125m python serve.py
open http://localhost:8080/docs
```

Finetune.

```sh
docker run -v $(pwd)/trained:/trained facebook-opt-125m python train.py ./sample-data/favorite-color-blue.jsonl
docker build -t facebook-opt-125m-trained -f ./trained.Dockerfile --build-arg=SRC_IMG=facebook-opt-125m .

# Run finetuned model (it is no good b/c the small dataset and epochs - I think).
docker run facebook-opt-125m-trained python infer.py "My favorite color is"
```
