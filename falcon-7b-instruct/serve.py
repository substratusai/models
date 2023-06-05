import torch
import uvicorn
from transformers import AutoTokenizer, AutoModelForCausalLM

from fastapi import FastAPI, Response
from pydantic import BaseModel

from infer import Inferer
from model import load_model, load_tokenizer

model = load_model()
tokenizer = load_tokenizer()
print("Model Generation Config:", model.generation_config)

device = "cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu"
inf = Inferer(model, tokenizer, device)

app = FastAPI()


class GenerationRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 200
    temperature: float = 0.8
    do_sample: bool = True
    top_k: int = 50
    top_p: float = 0.95


@app.get("/")
def root():
    return Response("Life is good. Checkout /docs for the API documentation.")


@app.post("/generate")
def generate_text(req: GenerationRequest):
    return {"generation": inf.infer(req)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
