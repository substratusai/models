# Download huggingface transformers and load and save the model and tokenizer to disk for later inference and finetuning.

from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "tiiuae/falcon-7b-instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_id, load_in_4bit=True, device_map="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)

save_path = '/built/'
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
