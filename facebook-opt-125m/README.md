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

Explore with a Notebook.

```sh
# Run Jupyter Notebook server.
docker run -p 8888:8888 facebook-opt-125m jupyter notebook --allow-root --ip=0.0.0.0 --NotebookApp.token='' --notebook-dir='/app'

# In another terminal: Open browser.
open http://localhost:8888
```

Run Server.

```sh
# Run longrunning HTTP API.
docker run -p 8080:8080 facebook-opt-125m python serve.py

# In another terminal: Open browser.
open http://localhost:8080/docs
```

Finetune.

```sh
# Run training job.
docker run -e DATA_PATH=/app/sample-data/favorite-color-blue.jsonl -v $(pwd)/ran:/ran -v $(pwd)/trained:/trained facebook-opt-125m jupyter nbconvert --debug --to notebook --execute train.ipynb --output /ran/train.ipynb

# Copy trained model into a new image.
docker build -t facebook-opt-125m-trained -f ./trained.Dockerfile --build-arg=SRC_IMG=facebook-opt-125m .

# Open a notebook to view the finetune job.
docker run -p 8888:8888 facebook-opt-125m-trained jupyter notebook --allow-root --ip=0.0.0.0 --NotebookApp.token='' --notebook-dir='/app'

# Notice the cells report the trained run...
# TODO: Improve this to show training graphs.
open http://localhost:8888/notebooks/train.ipynb

# Run finetuned model (it is no good b/c the small dataset and epochs - I think).
docker run facebook-opt-125m-trained python infer.py "My favorite color is"
```
