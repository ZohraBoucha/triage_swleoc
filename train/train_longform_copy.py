from transformers import LongformerForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM, EarlyStoppingCallback
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load and prepare data (as before)
df = pd.read_csv('/home/swleocresearch/Desktop/triage-ai/train/triage_train_balanced.csv', encoding='latin-1')
df = df[df['output'].isin(['surgery', 'discharge'])]
texts = df.iloc[:, 1].tolist()
labels = df.iloc[:, 2].tolist()

# Create label mapping
label_mapping = {label: idx for idx, label in enumerate(set(labels))}
labels = [label_mapping[label] for label in labels]

# Calculate class weights
class_counts = np.bincount(labels)
total_samples = len(labels)
class_weights = torch.tensor([total_samples / (len(class_counts) * count) for count in class_counts], dtype=torch.float)

print("Class weights:", class_weights)

# Split data, tokenize, etc. (as before)
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.1, random_state=42)

tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")

def tokenize_function(texts):
    return tokenizer(
        texts, 
        padding="max_length", 
        truncation=True, 
        max_length=4096,
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


# Modified compute_metrics function
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

# Load model
num_labels = len(set(labels))  # This should be 5 based on your class weights
model = LongformerForSequenceClassification.from_pretrained(
    "yikuan8/Clinical-Longformer",
    num_labels=num_labels
)
# Training arguments
training_args = TrainingArguments(
    output_dir='./results_medLF_balanced',
    evaluation_strategy="epoch",
    learning_rate= 3e-5, # increased from 2e-5
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs= 30 # increased from 15
    weight_decay=0.1, # increased from 0.01
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model= 'eval_accuracy', #'f1'
    # Add logging steps to see more frequent updates
    logging_steps=10,
    # Add evaluation steps to evaluate more frequently
    eval_steps=50,
    warmup_steps =500,
    lr_scheduler_type='linear',
    max_grad_norm=1.0,
)

# Instantiate custom Trainer with class weights
trainer = WeightedLossTrainer(
    class_weights=class_weights,
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# Train and evaluate
trainer.train()

# Save the model
model.save_pretrained('./medical_longformer_balanced')
tokenizer.save_pretrained('./medical_longformer_balanced')

# Final evaluation
final_metrics = trainer.evaluate()
print("Final Evaluation Metrics:", final_metrics)

# Extract metrics for plotting
epochs = []
train_loss = []
eval_loss = []
eval_accuracy = []
eval_f1 = []

for entry in trainer.state.log_history:
    if 'epoch' in entry:
        epochs.append(entry['epoch'])
        if 'loss' in entry:
            train_loss.append(entry['loss'])
        if 'eval_loss' in entry:
            eval_loss.append(entry['eval_loss'])
        if 'eval_accuracy' in entry:
            eval_accuracy.append(entry['eval_accuracy'])
        if 'eval_f1' in entry:
            eval_f1.append(entry['eval_f1'])

# # Plot training history
# plt.figure(figsize=(12,8))

# # Plot training loss if available
# if train_loss:
#     plt.plot(epochs[:len(train_loss)], train_loss, label='Training Loss')

# # Plot evaluation metrics
# if eval_loss:
#     plt.plot(epochs[:len(eval_loss)], eval_loss, label='Validation Loss')
# if eval_accuracy:
#     plt.plot(epochs[:len(eval_accuracy)], eval_accuracy, label='Validation Accuracy')
# if eval_f1:
#     plt.plot(epochs[:len(eval_f1)], eval_f1, label='Validation F1 Score')

# plt.title('Training History')
# plt.xlabel('Epoch')
# plt.ylabel('Metric Value')
# plt.legend()
# plt.savefig('training_history_medLF_balanced.png')
# plt.close()

# print("Training history plot saved as 'training_history_medLF_balanced.png'")