import pandas as pd 

import numpy as np 

import torch 

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, pipeline 

from datasets import Dataset, load_metric 

from sklearn.model_selection import train_test_split 

from imblearn.over_sampling import SMOTE 

  

# Load the quantized Llama 2 70B model 

llama_model_id = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4" 

llama_pipeline = pipeline( 

    "text-generation", 

    model=llama_model_id, 

    model_kwargs={"torch_dtype": torch.float16}, 

    device_map="auto", 

) 

  

def extract_content(text): 

    prompt = f""" 

    Extract the most relevant clinical information from the following referral letter.  

    Focus on key symptoms, diagnoses, and recommended actions. Summarize in about 200 words: 

  

    {text} 

  

    Summary: 

    """ 

     

    output = llama_pipeline( 

        prompt, 

        max_new_tokens=250, 

        do_sample=True, 

        temperature=0.7, 

        num_return_sequences=1 

    ) 

     

    return output[0]['generated_text'].split("Summary:")[-1].strip() 

  

# Load and preprocess data 

df = pd.read_csv('/home/swleocresearch/Desktop/triage-ai/datasets/triage_dataset.csv') 

df = df[df['output'].isin(['surgery', 'discharge'])].reset_index(drop=True) 

  

# Extract content using Llama 2 

df['extracted_content'] = df.iloc[:, 1].apply(extract_content) 

  

texts = df['extracted_content'].tolist() 

labels = df.iloc[:, 2].tolist() 

label_mapping = {label: idx for idx, label in enumerate(set(labels))} 

labels = [label_mapping[label] for label in labels] 

  

# Split the data 

train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.1, stratify=labels) 

  

# BERT Tokenizer and Model 

bert_tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT") 

bert_model = AutoModelForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", num_labels=len(label_mapping)) 

  

def tokenize_function(texts): 

    return bert_tokenizer(texts, padding="max_length", truncation=True, max_length=512) 

  

# Tokenize data 

train_encodings = tokenize_function(train_texts) 

val_encodings = tokenize_function(val_texts) 

  

# Apply SMOTE 

smote = SMOTE(random_state=42) 

train_features = np.column_stack((train_encodings['input_ids'], train_encodings['attention_mask'])) 

train_features_resampled, train_labels_resampled = smote.fit_resample(train_features, train_labels) 

  

# Convert back to tokenizer format 

train_encodings_resampled = { 

    'input_ids': train_features_resampled[:, :512].tolist(), 

    'attention_mask': train_features_resampled[:, 512:].tolist() 

} 

  

# Create datasets 

train_dataset = Dataset.from_dict({ 

    'input_ids': train_encodings_resampled['input_ids'], 

    'attention_mask': train_encodings_resampled['attention_mask'], 

    'label': train_labels_resampled 

}) 

  

val_dataset = Dataset.from_dict({ 

    'input_ids': val_encodings['input_ids'], 

    'attention_mask': val_encodings['attention_mask'], 

    'label': val_labels 

}) 

  

# Metrics 

metric = load_metric("glue", "mrpc") 

  

def compute_metrics(pred): 

    labels = pred.label_ids 

    preds = pred.predictions.argmax(-1) 

    precision, recall, f1, _ = metric.compute(predictions=preds, references=labels) 

    acc = (preds == labels).astype(np.float32).mean().item() 

    return { 

        "accuracy": acc, 

        "f1": f1, 

        "precision": precision, 

        "recall": recall 

    } 

  

# Training arguments 

training_args = TrainingArguments( 

    output_dir='./results', 

    evaluation_strategy="epoch", 

    learning_rate=2e-5, 

    per_device_train_batch_size=16, 

    per_device_eval_batch_size=16, 

    num_train_epochs=20, 

    weight_decay=0.01, 

    save_strategy="epoch", 

    save_total_limit=2, 

    load_best_model_at_end=True, 

    metric_for_best_model="f1" 

) 

  

# Trainer 

trainer = Trainer( 

    model=bert_model, 

    args=training_args, 

    train_dataset=train_dataset, 

    eval_dataset=val_dataset, 

    compute_metrics=compute_metrics 

) 

  

# Train the model 

trainer.train() 

  

# Evaluate 

eval_results = trainer.evaluate() 

print("Evaluation results:") 

print(eval_results) 

  

# Save the final model 

bert_model.save_pretrained('./final_model') 

bert_tokenizer.save_pretrained('./final_model') 