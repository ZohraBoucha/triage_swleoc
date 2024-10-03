from transformers import Trainer, TrainingArguments, AutoTokenizer
from datasets import Dataset, load_metric
import torch
from transformers import AutoModelForSequenceClassification
import pandas as pd
from sklearn.model_selection import train_test_split

# Load CSV file into a pandas DataFrame
df = pd.read_csv('/home/swleocresearch/Desktop/triage-ai/datasets/triage_dataset.csv')  # replace 'train.csv' with your CSV file name

df = df[df['output'].isin(['surgery', 'discharge'])]

# Reset the index if desired (optional)
df = df.reset_index(drop=True)

# Assuming the first column is the text and the second column is the label
texts = df.iloc[:, 1].tolist()  # First column as input texts
labels = df.iloc[:, 2].tolist()  # Second column as labels (currently strings)

# Create a label mapping (string to integer)
label_mapping = {label: idx for idx, label in enumerate(set(labels))}

# Convert string labels to integer labels
labels = [label_mapping[label] for label in labels]
print('labels: ', labels)

# Split the data
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.1)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

# Set a consistent max_length for all sequences
max_length = 512  # You can adjust this based on your dataset

# Tokenization function with consistent padding and truncation
def tokenize_function(texts):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=max_length)

# Tokenize training and validation data
train_encodings = tokenize_function(train_texts)
val_encodings = tokenize_function(val_texts)

# Convert into Dataset objects
train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask'],
    'label': train_labels  # Integer labels now
})

val_dataset = Dataset.from_dict({
    'input_ids': val_encodings['input_ids'],
    'attention_mask': val_encodings['attention_mask'],
    'label': val_labels  # Integer labels now
})

# Load the model with a classification head
model = AutoModelForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", num_labels=len(label_mapping))
for param in model.parameters():
    param.data = param.data.contiguous()
# Define the metric
metric = load_metric("accuracy")

def compute_metrics(pred):
    logits, labels = pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    return metric.compute(predictions=predictions, references=labels)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",  # 'eval_strategy' is correct in recent versions
    learning_rate=1e-6,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=500,
    weight_decay=0.01,
    save_strategy="epoch",  # Save the model after each epoch
    save_total_limit=2  # Only keep the last 2 models saved
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Save the trained model
model.save_pretrained('./saved_model')
tokenizer.save_pretrained('./saved_model')
