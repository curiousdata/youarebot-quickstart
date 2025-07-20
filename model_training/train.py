"""
pip install transformers datasets peft mlflow pandas scikit-learn
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
"""
import os
import json
import torch
import mlflow
import pandas as pd
from datetime import datetime
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from mlflow.models.signature import infer_signature
from mlflow.pyfunc import PythonModel
from dotenv import load_dotenv
import shutil

load_dotenv()

os.chdir(os.path.dirname(os.path.abspath(__file__)))


class ChatClassifier(PythonModel):
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def predict(self, context, model_input):
        if isinstance(model_input, str):
            texts = [model_input]
        else:
            texts = model_input.tolist() if hasattr(model_input, 'tolist') else list(model_input)

        inputs = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        return {
            "predictions": probs.argmax(dim=1).cpu().numpy(),
            "probabilities": probs.cpu().numpy()
        }


MODEL_NAME = "msg_cls"
MODEL_ALIAS = "champion"
mlflow.set_tracking_uri('http://127.0.0.1:5000')
EXPERIMENT_NAME = '/Text_Classification'
if not mlflow.get_experiment_by_name(EXPERIMENT_NAME):
    mlflow.create_experiment(EXPERIMENT_NAME)
mlflow.set_experiment(EXPERIMENT_NAME)

print(f"Tracking URI: {mlflow.get_tracking_uri()}")

def load_data():
    with open('train.json', 'r') as f:
        train_dialogs = json.load(f)

    train_labels = pd.read_csv('ytrain.csv')

    train_texts = []
    train_labels_list = []

    for dialog_id, messages in train_dialogs.items():
        dialog_labels = train_labels[train_labels['dialog_id'] == dialog_id]

        for message in messages:
            participant_index = int(message['participant_index'])  # Convert to int to match CSV

            label_row = dialog_labels[dialog_labels['participant_index'] == participant_index]

            if not label_row.empty:
                label = label_row['is_bot'].values[0]
                train_texts.append(message['text'])
                train_labels_list.append(label)

    with open('test.json', 'r') as f:
        test_dialogs = json.load(f)

    test_labels = pd.read_csv('ytest.csv')

    val_texts = []
    val_labels_list = []

    for dialog_id, messages in test_dialogs.items():
        dialog_ids = test_labels[test_labels['dialog_id'] == dialog_id]

        for message in messages:
            participant_index = int(message['participant_index'])

            if not dialog_ids[dialog_ids['participant_index'] == participant_index].empty:
                val_texts.append(message['text'])
                val_labels_list.append(0)

    train_dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels_list})
    val_dataset = Dataset.from_dict({'text': val_texts, 'label': val_labels_list})

    return DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })


def tokenize_function(examples, tokenizer):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def test_model_loading(model_path, tokenizer_path):
    """Test loading the saved model and running inference"""
    print("\nTesting model loading...")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2
    )

    peft_model = get_peft_model(base_model, LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=6,
        lora_alpha=8,
        lora_dropout=0.2,
        bias="none",
        target_modules=["query", "value"]
    ))
    peft_model.load_state_dict(torch.load(model_path))
    peft_model.eval()

    test_text = "This is a test message to classify."
    inputs = tokenizer(test_text, return_tensors="pt", truncation=True, padding=True, max_length=128)

    with torch.no_grad():
        outputs = peft_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        prediction = probs.argmax().item()

    print(f"\nTest text: '{test_text}'")
    print(f"Prediction: {prediction}")
    print(f"Probabilities: {probs.tolist()}")


def main():
    print('Start Training...')
    dataset = load_data()

    base_model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    tokenized_datasets = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=['text']
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=2
    )

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=6,
        lora_alpha=8,
        lora_dropout=0.2,
        bias="none",
        target_modules=["query", "value"]
    )

    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir='./logs',
        logging_steps=10,
        report_to="none"
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    with mlflow.start_run(run_name=f"{MODEL_NAME}-{MODEL_ALIAS}"):
        mlflow.log_params({
            "model_name": MODEL_NAME,
            "model_alias": MODEL_ALIAS,
            "base_model": base_model_name,
            "train_samples": len(tokenized_datasets['train']),
            "val_samples": len(tokenized_datasets['validation']),
            "lora_r": peft_config.r,
            "lora_alpha": peft_config.lora_alpha,
            "lora_dropout": peft_config.lora_dropout,
            "lora_target_modules": ",".join(peft_config.target_modules),
            "lora_bias": peft_config.bias
        })

        train_result = trainer.train()
        eval_metrics = trainer.evaluate()

        mlflow.log_metrics({
            "train_loss": train_result.training_loss,
            "eval_loss": eval_metrics["eval_loss"],
            "eval_accuracy": eval_metrics["eval_accuracy"],
            "eval_f1": eval_metrics["eval_f1"],
            "eval_precision": eval_metrics["eval_precision"],
            "eval_recall": eval_metrics["eval_recall"]
        })

        # Create model directory
        model_dir = ('models/model-%s' % datetime.now().isoformat()).replace(':', ' ').replace('.', ' ')
        os.makedirs(model_dir, exist_ok=True)

        # Save model as .pt file
        model_path = os.path.join(model_dir, "model.pt")
        torch.save(peft_model.state_dict(), model_path)

        # Save tokenizer
        tokenizer_path = os.path.join(model_dir, "tokenizer")
        tokenizer.save_pretrained(tokenizer_path)

        # Log artifacts
        mlflow.log_artifacts(model_dir)

        # Create and log pyfunc model
        pyfunc_model = ChatClassifier(tokenizer, peft_model)
        signature = infer_signature(["sample text"], pyfunc_model.predict(None, ["sample text"]))

        mlflow.pyfunc.log_model(
            python_model=pyfunc_model,
            name="model",
            registered_model_name=MODEL_NAME,
            signature=signature,
            pip_requirements=[
                "torch",
                "transformers",
                "peft",
                "datasets",
                "accelerate",
            ]
        )

        # Set model alias
        try:
            client = mlflow.tracking.MlflowClient()
            latest_version = client.get_latest_versions(MODEL_NAME, stages=["None"])[0].version
            client.set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, latest_version)
            print(f"Set alias '{MODEL_ALIAS}' for model '{MODEL_NAME}' version {latest_version}")
        except Exception as e:
            print(f"Error setting model alias: {str(e)}")

        print(f"Training complete. Model artifacts logged to: {model_dir}")
        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")

        # Demonstrate loading and using the saved model
        print("\nTesting model loading and inference:")
        # copy to classifier model
        shutil.copy(model_dir, os.path.join('../', 'api', 'services', 'classifier', 'model'))
        test_model_loading(model_path, tokenizer_path)


if __name__ == '__main__':
    main()