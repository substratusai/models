import transformers
from datasets import load_dataset
from torch.optim import Adam
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
import tqdm
import torch

import sys
from model import load_model, load_tokenizer

data_path = sys.argv[1]

tokenizer = load_tokenizer()
model = load_model(load_in_8bit=True)


lora_config2 = LoraConfig(
 r=8,
 lora_alpha=32,
 target_modules=["query_key_value"],
 lora_dropout=0.05,
 bias="none",
 task_type="CAUSAL_LM"
)

# prepare int-8 model for training
model = prepare_model_for_int8_training(model)

# add LoRA adaptor
model = get_peft_model(model, lora_config2)
model.print_trainable_parameters()

tokenizer.add_special_tokens({'pad_token': '[PAD]'})
def prompt(instruction, output):
    prompt = "{0}\n\n{1}\n{2}\n\n{3}\n{4}".format(
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.",
        ">>QUESTION<<",
        instruction,
        ">>ANSWER<<",
        output
    )
    return prompt

data = load_dataset("json", data_files=data_path)
print(data)
data = data.map(lambda x: tokenizer(prompt(x["prompt"], x["completion"]),
    max_length=1000, padding=True, truncation=True))
print("After tokenizing:", data)

trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="./outputs",
        optim="paged_adamw_8bit"
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

trainer.save_model("/trained")

device = "cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu"

text = "Write a YAML file to deploy a docker registry on K8s"

inputs = tokenizer(text, return_tensors="pt").to(device)
inputs.pop("token_type_ids")
outputs = model.generate(**inputs, max_new_tokens=300)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
