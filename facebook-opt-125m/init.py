# Import huggingface transformers and load and save the gpt2 model
# to a file.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "facebook/opt-350m"

model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

torch.save(model, '/model/model.pt')
torch.save(tokenizer, '/model/tokenizer.pt')

