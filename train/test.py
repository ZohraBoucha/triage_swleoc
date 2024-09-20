import pandas as pd 
import numpy as np 
import torch 
import os 
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments 
from datasets import Dataset, ClassLabel 
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report 
from sklearn.model_selection import train_test_split 
from transformers import pipeline 
from tqdm import tqdm 

# Load the quantized Llama 2 70B model 
llama_model_id = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4" 
llama_pipeline = pipeline( 
    "text-generation", 
    model=llama_model_id, 
    model_kwargs={"torch_dtype": torch.float16}, 
    device_map="auto", 
) 

def extract_content(text): 
    # Split very long texts into chunks of 8000 characters 
    chunks = [text[i:i+8000] for i in range(0, len(text), 8000)] 
    summaries = [] 

    for chunk in chunks: 
        prompt = f""" 
        Extract the most relevant clinical information from the following part of a referral letter.  
        Focus on key symptoms, diagnoses, and recommended actions. Summarize in about 100-150 words. 
        Ensure the summary is complete and ends with a full stop: 

        {chunk} 

        Summary: 
        """ 

        output = llama_pipeline( 
            prompt, 
            max_new_tokens=300,  # Increased from 150 to 300 
            do_sample=True, 
            temperature=0.7, 
            num_return_sequences=1 
        ) 

        summary = output[0]['generated_text'].split("Summary:")[-1].strip() 

        # Ensure the summary ends with a complete sentence 
        if not summary.endswith('.'): 
            last_sentence = re.split(r'(?<=[.!?])\s+', summary)[-1] 
            summary = summary[:summary.rfind(last_sentence)] 

        summaries.append(summary) 

    # Combine summaries if there were multiple chunks 
    final_summary = " ".join(summaries) 

    # If the combined summary is still too long, summarize it again 
    if len(final_summary) > 1000: 
        prompt = f""" 
        Summarize the following extracted information in about 200-250 words. 
        Ensure the summary is complete and ends with a full stop: 

        {final_summary} 

        Summary: 
        """ 

        output = llama_pipeline( 
            prompt, 
            max_new_tokens=400,  # Increased from 250 to 400 
            do_sample=True, 
            temperature=0.7, 
            num_return_sequences=1 
        ) 

        final_summary = output[0]['generated_text'].split("Summary:")[-1].strip() 

        # Ensure the final summary ends with a complete sentence 
        if not final_summary.endswith('.'): 
            last_sentence = re.split(r'(?<=[.!?])\s+', final_summary)[-1] 
            final_summary = final_summary[:final_summary.rfind(last_sentence)] 

    return final_summary  

# File paths 
input_csv_path = '/home/swleocresearch/Desktop/triage-ai/datasets/triage_dataset.csv' 
summarized_csv_path = '/home/swleocresearch/Desktop/triage-ai/datasets/summarized_dataset.csv' 

# Check if summarized dataset exists 
if os.path.exists(summarized_csv_path): 
    print("Loading pre-summarized dataset...") 
    df = pd.read_csv(summarized_csv_path) 
else: 
    print("Summarized dataset not found. Processing original dataset...") 
    # Load and preprocess data 
    df = pd.read_csv(input_csv_path) 
    df = df[df['output'].isin(['surgery', 'discharge'])].reset_index(drop=True) 

    # Create label mapping 
    label_mapping = {label: idx for idx, label in enumerate(df['output'].unique())} 
    reverse_label_mapping = {v: k for k, v in label_mapping.items()} 

    print("Extracting content from all letters...") 
    df['extracted_content'] = df.iloc[:, 1].apply(lambda x: extract_content(x)) 

    # Save the summarized dataset 
    df.to_csv(summarized_csv_path, index=False) 
    print(f"Summarized dataset saved to {summarized_csv_path}") 

# Create label mapping 
label_mapping = {label: idx for idx, label in enumerate(df['output'].unique())} 
reverse_label_mapping = {v: k for k, v in label_mapping.items()} 

texts = df['extracted_content'].tolist() 
labels = df['output'].tolist() 
label_ids = [label_mapping[label] for label in labels] 

# Split the data into train, validation, and test sets 
train_texts, temp_texts, train_labels, temp_labels = train_test_split(texts, label_ids, test_size=0.3, stratify=label_ids, random_state=42) 
val_texts, test_texts, val_labels, test_labels = train_test_split(temp_texts, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42) 

# BERT Tokenizer and Model 
bert_tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT") 
bert_model = AutoModelForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", num_labels=len(label_mapping)) 

def tokenize_function(texts): 
    return bert_tokenizer(texts, padding="max_length", truncation=True, max_length=512) 

# Tokenize data 
train_encodings = tokenize_function(train_texts) 
val_encodings = tokenize_function(val_texts) 
test_encodings = tokenize_function(test_texts) 

# Create datasets with proper label format 
def create_dataset(encodings, labels): 
    dataset = Dataset.from_dict({ 
        'input_ids': encodings['input_ids'], 
        'attention_mask': encodings['attention_mask'], 
        'label': labels 
    }) 

    # Convert label feature to ClassLabel 
    dataset = dataset.cast_column('label', ClassLabel(num_classes=len(label_mapping), names=list(label_mapping.keys()))) 
    return dataset 

train_dataset = create_dataset(train_encodings, train_labels) 
val_dataset = create_dataset(val_encodings, val_labels) 
test_dataset = create_dataset(test_encodings, test_labels) 

# Custom metric function 
def compute_metrics(pred): 
    labels = pred.label_ids 
    preds = pred.predictions.argmax(-1) 
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted') 
    acc = accuracy_score(labels, preds) 
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
    num_train_epochs=3, 
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

print("\nTraining the model...") 
trainer.train() 

# Evaluate on validation set 
print("\nEvaluating on validation set...") 
val_results = trainer.evaluate() 
print("Validation results:") 
print(val_results) 

# Predict on test set 
print("\nPredicting on test set...") 
test_predictions = trainer.predict(test_dataset) 
test_preds = test_predictions.predictions.argmax(-1) 

# Print detailed classification report for test set 
print("\nDetailed Test Set Performance:") 
print(classification_report(test_labels, test_preds, target_names=label_mapping.keys())) 

# Create a summary of predictions vs ground truth for test set 
print("\nDetailed Test Set Prediction Summary:") 
for i, (text, true_label, pred_label) in enumerate(zip(test_texts, test_labels, test_preds)): 
    true_label_str = reverse_label_mapping[true_label] 
    pred_label_str = reverse_label_mapping[pred_label] 
    print(f"\nTest Sample {i+1}:") 
    print(f"True Label: {true_label_str}") 
    print(f"Predicted Label: {pred_label_str}") 
    print(f"Prediction {'Correct' if true_label_str == pred_label_str else 'Incorrect'}") 
    print("Summarized Text:") 
    print(text[:500] + "..." if len(text) > 500 else text) 
    print("-" * 80) 

# Calculate overall test accuracy 
test_accuracy = accuracy_score(test_labels, test_preds) 
print(f"\nOverall Test Accuracy: {test_accuracy:.2f}") 

# Save the final model 
bert_model.save_pretrained('./final_model') 
bert_tokenizer.save_pretrained('./final_model') 