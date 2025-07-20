
import os
import torch
import mlflow.pyfunc
from dotenv import load_dotenv
from peft import get_peft_model, TaskType, LoraConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification
#
# load_dotenv()
#
# mlflow.set_tracking_uri("databricks")
print("\nTesting model loading...")

model_dir = os.path.join(os.getcwd(), 'models', 'model-test2')
model_path = os.path.join(model_dir, "model.pt")
tokenizer_path = os.path.join(model_dir, "tokenizer")
print(f"The file size is: {os.path.getsize(model_path)} bytes", model_path)

base_model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=2
)


peft_model = get_peft_model(base_model, LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    target_modules=["query", "value"]
))
peft_model.load_state_dict(torch.load(model_path))
peft_model.eval()

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)


for test_text in [
    "Привет",
    "Hola",
    "Hello",
    "Are you a bot?",
    "Hi there! How can I help you today? 😊",
    "ахах, прив гоу 5 вариантов",
]:
    inputs = tokenizer(test_text, return_tensors="pt", truncation=True, padding=True, max_length=128)

    with torch.no_grad():
        outputs = peft_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        prediction = probs.argmax().item()

    print(f"\nTest text: '{test_text}'")
    print(f"Prediction: {prediction}")
    print(f"Probabilities: {probs.tolist()[0][1]}")
