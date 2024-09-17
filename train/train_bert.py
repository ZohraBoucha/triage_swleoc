import optuna
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

# Define the metric
metric = load_metric("accuracy")

def compute_metrics(pred):
    logits, labels = pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    return metric.compute(predictions=predictions, references=labels)

# Optuna objective function for hyperparameter optimization
def objective(trial):
    # Hyperparameter suggestions by Optuna
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-7, 1e-5)
    num_train_epochs = trial.suggest_int('num_train_epochs', 3, 10)
    per_device_train_batch_size = trial.suggest_categorical('per_device_train_batch_size', [8, 16, 32])

    # Training arguments with hyperparameters from Optuna
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=8,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        save_strategy="epoch",
        save_total_limit=2,
        logging_dir='./logs',
        logging_steps=10
    )

    # Load the model with a classification head
    model = AutoModelForSequenceClassification.from_pretrained(
        "emilyalsentzer/Bio_ClinicalBERT", 
        num_labels=len(label_mapping)
    )

    for param in model.parameters():
        param.data = param.data.contiguous()

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

    # Evaluate the model after training
    eval_result = trainer.evaluate()

    # Return the evaluation accuracy to Optuna
    return eval_result['eval_accuracy']

# Create an Optuna study and store it in an SQLite database
study = optuna.create_study(
    direction="maximize", 
    storage="sqlite:///optuna_study.db",  # Save the study to an SQLite database
    study_name="triage_study",            # Name the study
    load_if_exists=True                   # Load existing study if it exists
)

# Optimize the study
study.optimize(objective, n_trials=10)

# Print the best trial and parameters
best_trial = study.best_trial
print(f"Best trial: Value: {best_trial.value}, Params: {best_trial.params}")
