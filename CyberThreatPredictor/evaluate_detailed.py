from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import classification_report
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# Load the model and tokenizer
model_path = '/content/drive/MyDrive/CyberThreatPredictor/saved_model_detailed'
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

# Load dataset
df = pd.read_csv('/content/drive/MyDrive/CyberThreatPredictor/dataset_detailed.csv')
labels = ["Social Engineering", "DDoS", "Zero-day Exploit", "SQL Injection", "Phishing", "Malware", "Ransomware"]
label_map = {label: idx for idx, label in enumerate(labels)}
df['label'] = df['label'].map(label_map)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'].tolist(), truncation=True, padding=True, max_length=512, return_tensors='pt')

class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Prepare the dataset
encodings = tokenize_function(df)
dataset = CustomDataset(encodings, df['label'].tolist())
dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

# Evaluate the model
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in dataloader:
        inputs = {key: batch[key].to(model.device) for key in batch if key != 'labels'}
        labels = batch['labels'].to(model.device)
        outputs = model(**inputs)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(classification_report(all_labels, all_preds, target_names=labels))
