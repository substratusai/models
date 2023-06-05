import sys
import torch

from model import load_model, load_tokenizer


class Inferer():
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def infer(self, prompt, max_new_tokens):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        # required due to issue with model.generate not supporting this
        inputs.pop("token_type_ids")
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    model = load_model()
    tokenizer = load_tokenizer()

    device = "cuda" if torch.cuda.is_available(
    ) else "mps" if torch.backends.mps.is_available() else "cpu"
    prompt = sys.argv[1]
    max_new_tokens = int(sys.argv[2]) if len(sys.argv) >= 3 else 400
    print(Inferer(model, tokenizer, device).infer(prompt, max_new_tokens))
