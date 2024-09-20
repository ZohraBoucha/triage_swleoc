import pandas as pd 
import numpy as np 
import torch 
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments 
from datasets import Dataset 
from sklearn.metrics import accuracy_score, precision_recall_fscore_support 
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

# Create label mapping 
label_mapping = {label: idx for idx, label in enumerate(df['output'].unique())} 
reverse_label_mapping = {v: k for k, v in label_mapping.items()} 

print("Extracting content from all letters...") 
extracted_contents = [] 
for index, row in tqdm(df.iterrows(), total=df.shape[0]): 
    original_text = row.iloc[1]  # Assuming the text is in the second column 
    original_label = row['output'] 
    print(f"\nProcessing Letter {index + 1}:") 
    print(f"Original Label: {original_label}") 
    print("Original Text:") 
    print(original_text[:500] + "..." if len(original_text) > 500 else original_text) 

    extracted = extract_content(original_text) 
    extracted_contents.append(extracted) 

    print("\nSummarized Text:") 
    print(extracted) 
    print("-" * 80) 

df['extracted_content'] = extracted_contents   

texts = df['extracted_content'].tolist() 
labels = df['output'].tolist() 
label_ids = [label_mapping[label] for label in labels] 

# BERT Tokenizer and Model 
bert_tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT") 
bert_model = AutoModelForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", num_labels=len(label_mapping)) 

def tokenize_function(texts): 
    return bert_tokenizer(texts, padding="max_length", truncation=True, max_length=512) 

# Tokenize all data 
encodings = tokenize_function(texts) 

# Create dataset 
dataset = Dataset.from_dict({ 
    'input_ids': encodings['input_ids'], 
    'attention_mask': encodings['attention_mask'], 
    'label': label_ids 
}) 

# Split the dataset 
train_dataset, eval_dataset = dataset.train_test_split(test_size=0.2, stratify_by_column="label").values() 

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
    eval_dataset=eval_dataset, 
    compute_metrics=compute_metrics 
) 

print("\nTraining the model...") 
trainer.train() 

# Evaluate 
eval_results = trainer.evaluate() 
print("Evaluation results:") 
print(eval_results) 

# Predict on the entire dataset 
print("\nMaking predictions on the entire dataset...") 
predictions = trainer.predict(dataset) 
predicted_labels = predictions.predictions.argmax(-1) 

# Create a summary of predictions vs ground truth 
print("\nDetailed Prediction Summary:") 
for i, (text, true_label, pred_label) in enumerate(zip(texts, label_ids, predicted_labels)): 
    true_label_str = reverse_label_mapping[true_label] 
    pred_label_str = reverse_label_mapping[pred_label] 
    print(f"\nLetter {i+1}:") 
    print(f"Original Label: {true_label_str}") 
    print(f"Predicted Label: {pred_label_str}") 
    print("Summarized Text:") 
    print(text[:500] + "..." if len(text) > 500 else text) 
    print("-" * 80) 

# Calculate overall accuracy 
accuracy = accuracy_score(label_ids, predicted_labels) 
print(f"\nOverall Accuracy: {accuracy:.2f}") 

# Save the final model 
bert_model.save_pretrained('./final_model') 
bert_tokenizer.save_pretrained('./final_model') 