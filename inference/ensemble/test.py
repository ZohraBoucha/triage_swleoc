import os
import pandas as pd
import re
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import gradio as gr
import logging
import numpy as np
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set max token length
MAX_LENGTH = 4096

# Base path for models
base_model_path = '/home/swleocresearch/Desktop/triage-ai/train/ensemble'

# Model configurations
model_configs = [
    {"name": "medical_longformer", "model": "yikuan8/Clinical-Longformer", "max_length": 4096},
    {"name": "medical_bert", "model": "emilyalsentzer/Bio_ClinicalBERT", "max_length": 512},
    {"name": "medical_roberta", "model": "allenai/biomed_roberta_base", "max_length": 512}
]

# Load the saved models and tokenizers
models = []
tokenizers = []
for config in model_configs:
    model_path = os.path.join(base_model_path, config["name"])
    if os.path.exists(model_path):
        models.append(AutoModelForSequenceClassification.from_pretrained(model_path))
        tokenizers.append(AutoTokenizer.from_pretrained(model_path))
    else:
        logger.warning(f"Model path not found: {model_path}. Skipping this model.")

if not models:
    raise ValueError("No models were successfully loaded. Please check the model paths.")

# Create a label mapping (string to integer)
label_mapping = {'discharge': 0, 'surgery': 1}
inv_label_mapping = {v: k for k, v in label_mapping.items()}


def extract_text_from_pdf(pdf_path):
    doc = DocumentFile.from_pdf(pdf_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    predictor = ocr_predictor(pretrained=True).to(device)
    result = predictor(doc)
    json_output = result.export()
    
    extracted_text = ""
    for page in json_output['pages']:
        for block in page['blocks']:
            for line in block['lines']:
                for word in line['words']:
                    extracted_text += word['value'] + ' '
                extracted_text += '\n'
    
    return extracted_text

def clean_text(text):
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters except periods and commas
    text = re.sub(r'[^\w\s.,]', '', text)
    # Convert to lowercase
    text = text.lower()
    return text.strip()

def tokenize_function(texts, tokenizer, max_length):
    return tokenizer(
        texts, 
        padding="max_length", 
        truncation=True, 
        max_length=max_length,
        return_tensors="pt"
    )

def clean_token(token):
    # Remove special tokens
    if token in ("[CLS]", "[SEP]", "<s>", "</s>"):
        return ""
    # Remove Ġ artifact from tokenization
    token = token.replace("Ġ", "")
    # Remove hashtags and join split words
    cleaned = re.sub(r'##', '', token)
    # Split merged words (simple heuristic, can be improved)
    cleaned = re.sub(r'([a-z])([A-Z])', r'\1 \2', cleaned)
    # Split numbers from words
    cleaned = re.sub(r'(\d+)([a-zA-Z])', r'\1 \2', cleaned)
    cleaned = re.sub(r'([a-zA-Z])(\d+)', r'\1 \2', cleaned)
    # Remove single letters (likely artifacts from tokenization)
    if len(cleaned) == 1 and cleaned.isalpha():
        return ""
    return cleaned.lower()

def get_feature_importance(chunk_tensor, attention_mask, model, tokenizer):
    # Get the attention weights
    with torch.no_grad():
        outputs = model(input_ids=chunk_tensor, attention_mask=attention_mask, output_attentions=True)
        attentions = outputs.attentions[-1].mean(dim=1).mean(dim=1).squeeze().cpu().numpy()
    
    # Decode tokens and combine their importance
    words = tokenizer.convert_ids_to_tokens(chunk_tensor[0])
    word_importance = defaultdict(float)
    current_word = ""
    for token, importance in zip(words, attentions):
        cleaned_token = clean_token(token)
        if cleaned_token:
            if token.startswith("Ġ") or (not current_word and cleaned_token):  # New word
                if current_word:
                    word_importance[current_word] += importance
                current_word = cleaned_token
            else:
                current_word += cleaned_token
        elif current_word:
            word_importance[current_word] += importance
            current_word = ""
    
    if current_word:
        word_importance[current_word] += importance
    
    # Sort words by importance
    sorted_words = sorted(word_importance.items(), key=lambda x: x[1], reverse=True)
    return sorted_words[:20]  # Return top 20 important words

import numpy as np

# Define weights for each model
model_weights = {
    "medical_longformer": 0,
    "medical_bert": 0,
    "medical_roberta": 1
}

def classify_text_ensemble(text):
    all_probabilities = []
    all_important_words = []
    model_names = []
    
    for model, tokenizer, config in zip(models, tokenizers, model_configs):
        model_names.append(config['name'])
        # Tokenize the entire text
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        # Process in chunks
        chunk_size = config['max_length'] - 2  # Account for [CLS] and [SEP] tokens
        chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
        
        chunk_probabilities = []
        chunk_important_words = []
        
        for chunk in chunks:
            # Add special tokens
            chunk = [tokenizer.cls_token_id] + chunk + [tokenizer.sep_token_id]
            chunk_tensor = torch.tensor([chunk])
            
            # Create attention mask
            attention_mask = torch.ones_like(chunk_tensor)
            
            # Set global attention on CLS token for Longformer
            if 'longformer' in config['name'].lower():
                global_attention_mask = torch.zeros_like(attention_mask)
                global_attention_mask[:, 0] = 1
                
                # Make predictions
                model.eval()
                with torch.no_grad():
                    outputs = model(input_ids=chunk_tensor, 
                                    attention_mask=attention_mask, 
                                    global_attention_mask=global_attention_mask)
            else:
                # Make predictions for other models
                model.eval()
                with torch.no_grad():
                    outputs = model(input_ids=chunk_tensor, 
                                    attention_mask=attention_mask)
            
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            chunk_probabilities.append(probabilities[0].tolist())
            
            # Get feature importance for this chunk
            important_words = get_feature_importance(chunk_tensor, attention_mask, model, tokenizer)
            chunk_important_words.extend(important_words)
        
        # Average probabilities across chunks
        avg_chunk_probabilities = np.mean(chunk_probabilities, axis=0)
        all_probabilities.append(avg_chunk_probabilities)
        
        # Combine important words from all chunks
        all_important_words.extend(chunk_important_words)

    # Apply weights to probabilities
    weighted_probabilities = []
    for probs, model_name in zip(all_probabilities, model_names):
        weight = model_weights.get(model_name, 1.0)  # Default weight is 1.0 if not specified
        weighted_probabilities.append(np.array(probs) * weight)

    # Sum the weighted probabilities
    final_probabilities = np.sum(weighted_probabilities, axis=0)
    # Normalize to ensure they sum to 1
    final_probabilities = final_probabilities / np.sum(final_probabilities)

    final_prediction = np.argmax(final_probabilities)
    
    # Get the predicted class
    predicted_class = inv_label_mapping[final_prediction]
    
    # Combine and sort important words from all models and chunks
    combined_important_words = sorted(all_important_words, key=lambda x: x[1], reverse=True)[:20]
    
    return predicted_class, final_probabilities, combined_important_words

# Update the post-processing function to handle weighted probabilities
def post_process_classification(predicted_class, probabilities, important_words):
    confidence_threshold = 0.55
    surgery_keywords = ['operation', 'surgical', 'procedure', 'incision', 'replacement', 'arthroscopy']
    discharge_keywords = ['home', 'followup', 'medication', 'instructions', 'release', 'conservative']

    if abs(probabilities[0] - probabilities[1]) < 0.1:  # If probabilities are very close
        # Check for surgery-specific keywords in the text
        surgery_score = sum(importance for word, importance in important_words if any(keyword in word.lower() for keyword in surgery_keywords))
        discharge_score = sum(importance for word, importance in important_words if any(keyword in word.lower() for keyword in discharge_keywords))
        
        if surgery_score > discharge_score:
            return 'surgery', probabilities, important_words
        elif discharge_score > surgery_score:
            return 'discharge', probabilities, important_words
        else:
            return "Uncertain", probabilities, important_words
    elif max(probabilities) < confidence_threshold:
        return "Uncertain", probabilities, important_words

    return predicted_class, probabilities, important_words

def process_pdf(pdf_file):
    # Extract text from PDF
    extracted_text = extract_text_from_pdf(pdf_file.name)
    
    if not extracted_text:
        return "Failed to extract text from PDF", ""
    
    # Clean the extracted text
    cleaned_text = clean_text(extracted_text)
    
    # Classify the cleaned text using the ensemble model
    predicted_class, avg_probabilities, important_words = classify_text_ensemble(cleaned_text)
    
    # Post-process the classification
    final_class, avg_probabilities, important_words = post_process_classification(predicted_class, avg_probabilities, important_words)
    
    # Prepare the output
    result = f"Predicted class: {final_class}\n\nProbabilities:\n"
    for label, prob in zip(label_mapping.keys(), avg_probabilities):
        result += f"{label}: {prob:.4f}\n"
    
    result += "\nTop 20 Important words:\n"
    for word, importance in important_words:
        result += f"{word}: {importance:.4f}\n"
    
    # Highlight important words in the text
    highlighted_text = extracted_text
    for word, _ in important_words:
        highlighted_text = re.sub(r'\b' + re.escape(word) + r'\b', f'<span style="background-color: yellow;">{word}</span>', highlighted_text, flags=re.IGNORECASE)
    
    return result, gr.HTML(highlighted_text)

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Medical Referral Triage Classifier (Ensemble Model)")
    gr.Markdown("Upload a medical referral PDF to classify it into one of the triage categories: discharge or surgery.")
    
    with gr.Row():
        pdf_input = gr.File(label="Upload PDF")
    
    with gr.Row():
        classify_btn = gr.Button("Classify")
    
    with gr.Row():
        result_output = gr.Textbox(label="Classification Result")
        text_output = gr.HTML(label="Extracted Text with Highlighted Important Words")
    
    classify_btn.click(process_pdf, inputs=pdf_input, outputs=[result_output, text_output])

# Launch the interface
demo.launch()