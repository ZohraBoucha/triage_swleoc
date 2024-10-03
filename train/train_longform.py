from transformers import LongformerForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer, EarlyStoppingCallback
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from torch.utils.data import WeightedRandomSampler, DataLoader
from transformers import TrainerCallback
from collections import Counter

# Load and prepare data
df = pd.read_csv('/home/swleocresearch/Desktop/triage-ai/datasets/triage_dataset_train.csv', encoding='latin-1')
df = df[df['output'].isin(['surgery', 'discharge', 'injection', 'fu', 'physio'])]
texts = df.iloc[:, 1].tolist()
labels = df.iloc[:, 2].tolist()

print(f"Total number of samples in the dataset: {len(df)}")
print(f"Number of samples per class: {df['output'].value_counts()}")

# Create label mapping
label_mapping = {label: idx for idx, label in enumerate(set(labels))}
labels = [label_mapping[label] for label in labels]

# Calculate class weights
class_counts = Counter(labels)
total_samples = len(labels)
class_weights = torch.tensor([total_samples / (len(class_counts) * count) for count in class_counts.values()], dtype=torch.float)

print("Class weights:", class_weights)

tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")

def tokenize_function(texts):
    return tokenizer(
        texts, 
        padding="max_length", 
        truncation=True, 
        max_length=4096,
        return_tensors="pt"
    )

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

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        train_sampler = WeightedRandomSampler(
            weights=self.class_weights,
            num_samples=len(self.train_dataset),
            replacement=True
        )
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=0,
            pin_memory=self.args.dataloader_pin_memory,
        )

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

class ClassDistributionCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if 'train_dataloader' in kwargs:
            labels = [batch['labels'] for batch in kwargs['train_dataloader']]
            labels = torch.cat(labels).cpu().numpy()
            unique, counts = np.unique(labels, return_counts=True)
            print(f"Epoch {state.epoch}: Class distribution in training data:")
            for label, count in zip(unique, counts):
                print(f"  Class {label}: {count}")

# StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_index, val_index) in enumerate(skf.split(texts, labels)):
    print(f"Fold {fold + 1}")
    
    train_texts = [texts[i] for i in train_index]
    train_labels = [labels[i] for i in train_index]
    val_texts = [texts[i] for i in val_index]
    val_labels = [labels[i] for i in val_index]

    train_encodings = tokenize_function(train_texts)
    val_encodings = tokenize_function(val_texts)

    train_dataset = Dataset.from_dict({
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask'],
        'labels': train_labels
    })

    val_dataset = Dataset.from_dict({
        'input_ids': val_encodings['input_ids'],
        'attention_mask': val_encodings['attention_mask'],
        'labels': val_labels
    })

    model = LongformerForSequenceClassification.from_pretrained(
        "yikuan8/Clinical-Longformer",
        num_labels=len(label_mapping)
    )

    training_args = TrainingArguments(
        output_dir=f'./results_medLF_fold_{fold + 1}',
        evaluation_strategy="steps",
        eval_steps=20,
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=10,
        weight_decay=0.01,
        save_strategy="steps",
        save_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        logging_steps=10,
    )

    trainer = WeightedLossTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3), ClassDistributionCallback()]
    )

    try:
        trainer.train()
    except Exception as e:
        print(f"An error occurred during training: {e}")
        continue

    # Save the model for this fold
    model.save_pretrained(f'./medical_longformer_fold_{fold + 1}')
    tokenizer.save_pretrained(f'./medical_longformer_fold_{fold + 1}')

    # Final evaluation for this fold
    final_metrics = trainer.evaluate()
    print(f"Final Evaluation Metrics for Fold {fold + 1}:", final_metrics)

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

    # Plot training history
    plt.figure(figsize=(12,8))

    if train_loss:
        plt.plot(epochs[:len(train_loss)], train_loss, label='Training Loss')
    if eval_loss:
        plt.plot(epochs[:len(eval_loss)], eval_loss, label='Validation Loss')
    if eval_accuracy:
        plt.plot(epochs[:len(eval_accuracy)], eval_accuracy, label='Validation Accuracy')
    if eval_f1:
        plt.plot(epochs[:len(eval_f1)], eval_f1, label='Validation F1 Score')

    plt.title(f'Training History - Fold {fold + 1}')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.savefig(f'training_history_medLF_fold_{fold + 1}.png')
    plt.close()

    print(f"Training history plot saved as 'training_history_medLF_fold_{fold + 1}.png'")

print("Training completed for all folds.")