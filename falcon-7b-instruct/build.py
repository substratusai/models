# Download huggingface transformers and load and save the model and tokenizer to disk for later inference and finetuning.

from huggingface_hub import snapshot_download

model_id = "tiiuae/falcon-7b-instruct"

snapshot_download(repo_id=model_id, local_dir="/built", local_dir_use_symlinks=False,
        ignore_patterns="*weight.bin")
