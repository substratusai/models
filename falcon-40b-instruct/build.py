import os
from huggingface_hub import snapshot_download

model_id = os.getenv("MODEL_ID")

snapshot_download(repo_id=model_id, local_dir="/built", local_dir_use_symlinks=False,
        ignore_patterns="*weight.bin")
