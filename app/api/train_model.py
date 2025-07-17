import pandas as pd
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from datasets import Dataset

# === Resolve paths relative to this script ===
CURRENT_DIR = Path(__file__).resolve().parent
train_json_path = CURRENT_DIR / "train.json"
ytrain_csv_path = CURRENT_DIR / "ytrain.csv"
model_dir = CURRENT_DIR / "model"

# === Load and merge data ===
with open(train_json_path, "r") as f:
    train_json = json.load(f)

ytrain_df = pd.read_csv(ytrain_csv_path)
train_df = pd.DataFrame(train_json)
df = pd.merge(train_df, ytrain_df, on="id")

# === Preprocess ===
df = df.rename(columns={"last_user_message": "text", "is_bot": "label"})
df["label"] = df["label"].astype(int)

# === Tokenization ===
MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding=True)

dataset = Dataset.from_pandas(df[["text", "label"]])
dataset = dataset.train_test_split(test_size=0.1)
dataset = dataset.map(tokenize, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# === Model ===
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# === Training ===
training_args = TrainingArguments(
    output_dir=str(CURRENT_DIR / "results"),
    evaluation_strategy="epoch",
    save_strategy="no",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    logging_dir=str(CURRENT_DIR / "logs"),
    load_best_model_at_end=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
)

trainer.train()

# === Save model ===
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)

print(f"✅ Model and tokenizer saved to: {model_dir}")