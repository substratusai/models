# Model: Facebook Opt (125 million Params)

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

Explore and develop with a Notebook.

```sh
# Run a Jupyter Notebook.
docker run -it -v $(pwd)/src:/model/src -p 8888:8888 facebook-opt-125m dev.sh

# In another terminal: Open browser.
open http://localhost:8888

# Now you can edit the contents of `src/`.
```


Run Server.

```sh
# Run longrunning HTTP API.
docker run -it -p 8080:8080 facebook-opt-125m serve.sh

# In another terminal: Open browser.
open http://localhost:8080/docs
```

Finetune.

```sh
# Run training job.
docker run -e DATA_PATH=/model/hack/sample-data.jsonl -v $(pwd)/logs:/model/logs -v $(pwd)/trained:/model/trained facebook-opt-125m train.sh

# Build a new image from the trained model.
docker build -t facebook-opt-125m-trained -f ./hack/trained.Dockerfile --build-arg=SRC_IMG=facebook-opt-125m .

# Open a notebook to view the finetune job.
docker run -p 8888:8888 facebook-opt-125m-trained jupyter notebook --allow-root --ip=0.0.0.0 --NotebookApp.token='' --notebook-dir='/model'

# Notice the cells report the trained run...
# TODO: Improve this to show training graphs.
open http://localhost:8888/notebooks/logs/train.ipynb

# Run finetuned model (it is no good b/c the small dataset and epochs - I think).
docker run facebook-opt-125m-trained python ./src/infer.py "My favorite color is"
```
