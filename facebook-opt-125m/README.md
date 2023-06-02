# Model: Facebook Opt-125m

Build.

```sh
docker build -t facebook-opt-125m .
```

Run inference.

```sh
docker run facebook-opt-125m python infer.py "My favorite color is"
```

Run Notebook.

```sh
docker run -p 8888:8888 facebook-opt-125m jupyter notebook --allow-root --ip=0.0.0.0 --NotebookApp.token='' --notebook-dir='/app'
open localhost:8888
```

Run Server.

```sh
docker run -p 8080:8080 facebook-opt-125m python serve.py
open localhost:8888
```

Finetune.

```sh
docker run facebook-opt-125m python train.py ./data/sample/favorite-color-blue.jsonl
```
