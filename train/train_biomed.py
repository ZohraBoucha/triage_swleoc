import os
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer, TrainerCallback
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from transformers import get_linear_schedule_with_warmup

# Load and prepare data
df = pd.read_csv('/home/swleocresearch/Desktop/triage-ai/train/triage_train_balanced.csv', encoding='latin-1')
df = df[df['output'].isin(['surgery', 'discharge'])]
texts = df.iloc[:, 1].tolist()
labels = df.iloc[:, 2].tolist()

# Create label mapping
label_mapping = {label: idx for idx, label in enumerate(set(labels))}
labels = [label_mapping[label] for label in labels]

# Calculate class weights
class_counts = torch.bincount(torch.tensor(labels))
total_samples = len(labels)
class_weights = torch.tensor([total_samples / (len(class_counts) * count) for count in class_counts], dtype=torch.float)

print("Class weights:", class_weights)

# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.1, random_state=42)

# Load tokenizer and model
model_name = "dmis-lab/biobert-v1.1"  # Changed to BioBERT
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(set(labels)))

def tokenize_function(texts):
    return tokenizer(
        texts, 
        padding="max_length", 
        truncation=True, 
        max_length=512,
        return_tensors="pt"
    )

train_encodings = tokenize_function(train_texts)
val_encodings = tokenize_function(val_texts)

train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'].tolist(),
    'attention_mask': train_encodings['attention_mask'].tolist(),
    'labels': train_labels
})

val_dataset = Dataset.from_dict({
    'input_ids': val_encodings['input_ids'].tolist(),
    'attention_mask': val_encodings['attention_mask'].tolist(),
    'labels': val_labels
})

# Custom Trainer with weighted loss
class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# Compute metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'mcc': mcc,
    }

# Custom EarlyStopping callback
class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, early_stopping_patience: int = 3, early_stopping_threshold: float = 0.0):
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.early_stopping_patience_counter = 0
        self.best_metric = None

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        eval_metric = metrics.get("eval_loss")
        if self.best_metric is None:
            self.best_metric = eval_metric
        elif eval_metric > self.best_metric + self.early_stopping_threshold:
            self.early_stopping_patience_counter += 1
            if self.early_stopping_patience_counter >= self.early_stopping_patience:
                control.should_training_stop = True
        else:
            self.best_metric = eval_metric
            self.early_stopping_patience_counter = 0

# Training arguments
training_args = TrainingArguments(
    output_dir='./results_bioBERT_balanced',
    evaluation_strategy="epoch",
    save_strategy="no",
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=20,
    weight_decay=0.03,
    load_best_model_at_end=False,
    metric_for_best_model='eval_loss',
    logging_steps=10,
    eval_steps=50,
    warmup_steps=1000,
    lr_scheduler_type="linear",
    max_grad_norm=1.0,
)

# Instantiate custom Trainer
trainer = WeightedLossTrainer(
    class_weights=class_weights,
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)

# Train and evaluate
trainer.train()

# Save the final model
output_dir = './biobert_balanced_final'
os.makedirs(output_dir, exist_ok=True)
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

# Final evaluation
final_metrics = trainer.evaluate()
print("Final Evaluation Metrics:", final_metrics)

# Plot training history
# (Keep the plotting code as before)