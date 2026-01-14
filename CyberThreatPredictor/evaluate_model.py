import os
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import evaluate

# Set paths
model_directory = '/content/drive/MyDrive/CyberThreatPredictor/saved_model'
dataset_path = '/content/drive/MyDrive/CyberThreatPredictor/dataset.csv'

# Load dataset
dataset = load_dataset('csv', data_files={'test': dataset_path})

# Load metrics for evaluation
accuracy_metric = evaluate.load('accuracy')
precision_metric = evaluate.load('precision')
recall_metric = evaluate.load('recall')
f1_metric = evaluate.load('f1')

# Load the model and tokenizer from the saved model directory
model = DistilBertForSequenceClassification.from_pretrained(model_directory)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_directory)

# Tokenize the test dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_test_dataset = dataset['test'].map(tokenize_function, batched=True)

# Define compute_metrics function for additional metrics
def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(axis=1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    precision = precision_metric.compute(predictions=predictions, references=labels, average='binary')
    recall = recall_metric.compute(predictions=predictions, references=labels, average='binary')
    f1 = f1_metric.compute(predictions=predictions, references=labels, average='binary')

    return {
        'accuracy': accuracy['accuracy'],
        'precision': precision['precision'],
        'recall': recall['recall'],
        'f1': f1['f1'],
    }

# Define evaluation arguments
training_args = TrainingArguments(
    output_dir='./results',
    per_device_eval_batch_size=8,
    evaluation_strategy="no",
    logging_dir='./logs',
    logging_steps=10,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=tokenized_test_dataset,
    compute_metrics=compute_metrics
)

# Evaluate the model
eval_results = trainer.evaluate()

# Print the evaluation results
print(f"Evaluation results: {eval_results}")
