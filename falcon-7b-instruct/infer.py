import sys
import torch

from model import load_model, load_tokenizer


class Inferer():
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def infer(self, generation_request):
        # prompt_template = f"### Instruction: {prompt}\n### Response:"
        inputs = self.tokenizer(generation_request.prompt, return_tensors="pt").to(self.device)
        # required due to issue with model.generate not supporting this
        inputs.pop("token_type_ids")
        generation_request = generation_request.dict(exclude={"prompt"})
        outputs = self.model.generate(**inputs, **generation_request)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
