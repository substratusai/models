from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "/built/"

def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    return model

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    return tokenizer

