from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import os
import shutil

# Load dataset
dataset_path = '/content/drive/MyDrive/CyberThreatPredictor/dataset.csv'
dataset = load_dataset('csv', data_files={'train': dataset_path, 'test': dataset_path})

# Initialize tokenizer and model
model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir='/content/drive/MyDrive/CyberThreatPredictor/model',
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
)

# Train the model
trainer.train()

# Save model and tokenizer
model_dir = '/content/drive/MyDrive/CyberThreatPredictor/model'
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)

# Optional: Copy saved model to another location in Google Drive
saved_model_dir = '/content/drive/MyDrive/CyberThreatPredictor/saved_model'
if not os.path.exists(saved_model_dir):
    os.makedirs(saved_model_dir)
shutil.copytree(model_dir, saved_model_dir, dirs_exist_ok=True)
