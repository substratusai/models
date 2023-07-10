from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "/built/"

def load_model(load_in_8bit: bool = False):
    if load_in_8bit == False:
        return AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    else:
        return AutoModelForCausalLM.from_pretrained(
            model_path, load_in_8bit=True, device_map={"":0}, trust_remote_code=True)


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    return tokenizer

