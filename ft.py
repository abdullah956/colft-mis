!pip install transformers datasets peft trl accelerate bitsandbytes -q

from huggingface_hub import login
login("")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "mistralai/Mistral-7B-v0.1"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype="float16",
)

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

from datasets import Dataset

data = {
    "prompt": [
        "What is the capital of France?",
        "Explain the water cycle.",
        "Who wrote Hamlet?",
    ],
    "response": [
        "The capital of France is Paris.",
        "The water cycle describes how water moves through the Earth: evaporation, condensation, precipitation, and collection.",
        "William Shakespeare wrote Hamlet."
    ]
}

dataset = Dataset.from_dict(data)

def preprocess(batch):
    texts = [
        f"QUESTION: {q}\nANSWER: {a}"
        for q, a in zip(batch["prompt"], batch["response"])
    ]
    tokenized = tokenizer(texts, max_length=256, truncation=True, padding="max_length")
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

dataset = dataset.map(preprocess, batched=True, remove_columns=["prompt", "response"])

# Apply PEFT / LoRA
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=8, lora_alpha=16, target_modules=["q_proj","k_proj","v_proj","o_proj"],
    lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

import os
os.environ["WANDB_DISABLED"] = "true"

# Training using transformers.Trainer
from transformers import Trainer, TrainingArguments
import torch

training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_steps=50,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=1,
    output_dir="./mistral-finetuned",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=lambda data: {
        "input_ids": torch.stack([torch.tensor(f["input_ids"]) for f in data]),
        "attention_mask": torch.stack([torch.tensor(f["attention_mask"]) for f in data]),
        "labels": torch.stack([torch.tensor(f["labels"]) for f in data])
    }
)

trainer.train()
trainer.save_model("./mistral-finetuned")
tokenizer.save_pretrained("./mistral-finetuned")

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype="float16"
)

model = AutoModelForCausalLM.from_pretrained("./mistral-finetuned", quantization_config=bnb_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("./mistral-finetuned")

input_text = "### Prompt:\nWho wrote Hamlet?\n\n### Response:\n"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)

output_ids = model.generate(input_ids, max_new_tokens=50)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))

!zip -r mistral-finetuned.zip ./mistral-finetuned


from google.colab import files
files.download("mistral-finetuned.zip")
