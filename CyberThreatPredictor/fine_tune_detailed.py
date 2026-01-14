from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import os

# Load dataset
df = pd.read_csv('/content/drive/MyDrive/CyberThreatPredictor/dataset_detailed.csv')

# Define labels
labels = ["Social Engineering", "DDoS", "Zero-day Exploit", "SQL Injection", "Phishing", "Malware", "Ransomware"]
label_map = {label: idx for idx, label in enumerate(labels)}

# Convert labels to indices
df['label'] = df['label'].map(label_map)

# Split dataset
train_texts, val_texts, train_labels, val_labels = train_test_split(df['text'].tolist(), df['label'].tolist(), test_size=0.2)

# Initialize tokenizer and model
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(labels))

# Tokenize data
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

# Create datasets
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = CustomDataset(train_encodings, train_labels)
val_dataset = CustomDataset(val_encodings, val_labels)

# Training arguments
training_args = TrainingArguments(
    output_dir='/content/drive/MyDrive/CyberThreatPredictor/saved_model_detailed',  # Path to save the new fine-tuned model
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",  # Ensure the model is saved every epoch
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train and evaluate
trainer.train()

# Save the model and tokenizer
model_save_path = '/content/drive/MyDrive/CyberThreatPredictor/saved_model_detailed'
tokenizer.save_pretrained(model_save_path)
trainer.save_model(model_save_path)
