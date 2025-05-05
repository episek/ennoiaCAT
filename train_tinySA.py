import os
import torch
import numpy as np
import json
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
from sklearn.metrics import accuracy_score

# === Configuration ===
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
dataset_path = "./tinySA_train.json"
save_dir = "./tinyllama_tinysa_lora"

# === Check for resumed training ===
adapter_exists = any(
    os.path.isfile(os.path.join(save_dir, f))
    for f in ["adapter_model.safetensors", "adapter_model.bin"]
)
resume_training = adapter_exists

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# === Show CUDA Device Info ===
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")

# === Load and Process Dataset ===
data = load_dataset("json", data_files=dataset_path)

def format_instruction(example):
    return {
        "text": f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['response']}"
    }

data = data.map(format_instruction)
data = data["train"].train_test_split(test_size=0.1)
train_data = data["train"]
eval_data = data["test"]
print("Dataset loaded and split into train and eval sets.")

# === Tokenization ===
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=512)

train_data = train_data.map(tokenize_function, batched=True)
eval_data = eval_data.map(tokenize_function, batched=True)
print("Tokenization complete.")

# === Quantization Config ===
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# === Load Model and Apply LoRA ===
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16
)
base_model = prepare_model_for_kbit_training(base_model)

if resume_training:
    print("🔄 Resuming training from previous LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, save_dir)
else:
    print("🆕 Starting new training with fresh LoRA adapter...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base_model, lora_config)

print("Model loaded with LoRA configuration.")

# === Training Arguments ===
training_args = TrainingArguments(
    output_dir=save_dir,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    logging_dir="./logs",
    num_train_epochs=3,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    bf16=torch.cuda.is_available(),
    save_total_limit=1,
    push_to_hub=False,
    report_to="none"
)

# === Evaluation Metric Function ===
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    preds = np.argmax(logits, axis=-1)

    mask = labels != -100
    correct = (preds == labels) & mask
    accuracy = correct.sum() / mask.sum()

    return {"accuracy": accuracy.item()}

# === Trainer Initialization ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    compute_metrics=compute_metrics
)

print("Trainer initialized.")

# === Start Training with Optimizer Fallback ===
try:
    trainer.train(resume_from_checkpoint=resume_training)
except ValueError as e:
    if "parameter group that doesn't match" in str(e):
        print("⚠️ Optimizer state mismatch. Retrying without loading optimizer state.")
        trainer.train(resume_from_checkpoint=False)
    else:
        raise
print("Training complete.")


# === Final Evaluation and Feedback ===
eval_results = trainer.evaluate()
final_loss = eval_results.get("eval_loss", None)
print(f"\n📊 Final Evaluation Loss: {final_loss:.4f}")

if final_loss is not None:
    if final_loss < 1.0:
        print("✅ Training looks excellent! Loss is very low.")
    elif final_loss < 2.0:
        print("👍 Training is good. Model is learning reasonably well.")
    elif final_loss < 3.0:
        print("⚠️ Training is okay, but could be improved. Consider tuning hyperparameters.")
    else:
        print("❌ Training may have issues (e.g., underfitting or noisy data). Review setup.")
else:
    print("❓ Could not compute final loss.")

# === Save Evaluation Results ===
with open(os.path.join(save_dir, "final_eval_results.json"), "w") as f:
    json.dump(eval_results, f, indent=4)
print(f"📁 Evaluation results saved to {save_dir}/final_eval_results.json")

# === Save Model and Tokenizer ===
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print("Model and tokenizer saved.")

# === Verify Adapter Files ===
required_config = "adapter_config.json"
required_model = None
if os.path.isfile(os.path.join(save_dir, "adapter_model.safetensors")):
    required_model = "adapter_model.safetensors"
elif os.path.isfile(os.path.join(save_dir, "adapter_model.bin")):
    required_model = "adapter_model.bin"

if not required_model or not os.path.isfile(os.path.join(save_dir, required_config)):
    raise FileNotFoundError("❌ Missing required adapter files after training.")
else:
    print(f"✅ Training complete. Adapter saved as '{required_model}' with config.")
