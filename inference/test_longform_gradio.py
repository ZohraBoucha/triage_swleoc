import os
import pandas as pd
import re
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import torch
from transformers import LongformerTokenizer, LongformerForSequenceClassification
import gradio as gr
import logging
import numpy as np
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set max token length
MAX_LENGTH = 4096

# Load the saved model and tokenizer
model_path = '/home/swleocresearch/Desktop/triage-ai/train/medical_longformer_balanced'
model = LongformerForSequenceClassification.from_pretrained(model_path)
tokenizer = LongformerTokenizer.from_pretrained(model_path)

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

def tokenize_function(texts):
    return tokenizer(
        texts, 
        padding="max_length", 
        truncation=True, 
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

def get_feature_importance(text, predicted_class, probabilities):
    # Tokenize the text
    encoding = tokenizer(text, return_tensors="pt")
    input_ids = encoding["input_ids"]
    
    # Get the attention weights for the predicted class
    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_attentions=True)
        attentions = outputs.attentions[-1].mean(dim=1).mean(dim=1).squeeze().cpu().numpy()
    
    # Decode tokens and combine their importance
    words = tokenizer.convert_ids_to_tokens(input_ids[0])
    word_importance = defaultdict(float)
    current_word = ""
    for token, importance in zip(words, attentions):
        if token.startswith("Ġ"):  # New word in RoBERTa tokenizer
            current_word = token[1:]
        else:
            current_word += token.replace("Ġ", "")
        word_importance[current_word] += importance
    
    # Sort words by importance
    sorted_words = sorted(word_importance.items(), key=lambda x: x[1], reverse=True)
    return sorted_words[:20]  # Return top 20 important words

# def classify_text(text):
#     # Tokenize the text
#     encodings = tokenize_function([text])

#     # Set global attention on CLS token
#     global_attention_mask = torch.zeros_like(encodings['attention_mask'])
#     global_attention_mask[:, 0] = 1

#     # Make predictions
#     model.eval()
#     with torch.no_grad():
#         outputs = model(
#             input_ids=encodings['input_ids'],
#             attention_mask=encodings['attention_mask'],
#             global_attention_mask=global_attention_mask
#         )
#         predictions = torch.argmax(outputs.logits, dim=-1)
#         probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

#     predicted_class = inv_label_mapping[predictions.item()]
#     probabilities = probabilities[0].tolist()

#     # Get feature importance
#     important_words = get_feature_importance(text, predicted_class, probabilities)

#     return predicted_class, probabilities, important_words

def classify_text_in_chunks(text, max_length=MAX_LENGTH):
    # Tokenize the entire text
    tokens = tokenizer.encode(text, truncation=False)
    
    # Split tokens into chunks of max_length
    chunk_size = max_length - 2  # account for special tokens
    token_chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    
    # Initialize lists to store results
    all_probabilities = []
    all_predictions = []
    all_important_words = []
    
    for chunk in token_chunks:
        # Rebuild text from the chunked tokens and tokenize it for the model
        chunk_text = tokenizer.decode(chunk, clean_up_tokenization_spaces=True)
        encodings = tokenize_function([chunk_text])
        
        # Set global attention on CLS token for the chunk
        global_attention_mask = torch.zeros_like(encodings['attention_mask'])
        global_attention_mask[:, 0] = 1

        # Make predictions
        model.eval()
        with torch.no_grad():
            outputs = model(
                input_ids=encodings['input_ids'],
                attention_mask=encodings['attention_mask'],
                global_attention_mask=global_attention_mask
            )
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
            all_probabilities.append(probabilities)
            all_predictions.append(np.argmax(probabilities))
        
        # Get feature importance for this chunk
        important_words = get_feature_importance(chunk_text, np.argmax(probabilities), probabilities)
        all_important_words.extend(important_words)

    # Average probabilities across all chunks
    avg_probabilities = np.mean(all_probabilities, axis=0)
    final_prediction = np.argmax(avg_probabilities)
    
    # Get the predicted class
    predicted_class = inv_label_mapping[final_prediction]
    
    # Combine important words from all chunks
    combined_important_words = sorted(all_important_words, key=lambda x: x[1], reverse=True)[:20]
    
    return predicted_class, avg_probabilities, combined_important_words


def post_process_classification(predicted_class, probabilities, important_words):
    confidence_threshold = 0.55  # Lowered from 0.6
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

# def process_pdf(pdf_file):
#     # Extract text from PDF
#     extracted_text = extract_text_from_pdf(pdf_file.name)
    
#     if not extracted_text:
#         return "Failed to extract text from PDF", ""
    
#     # Clean the extracted text
#     cleaned_text = clean_text(extracted_text)
    
#     # Classify the cleaned text
#     predicted_class, probabilities, important_words = classify_text(cleaned_text)
    
#     # Post-process the classification
#     final_class, probabilities, important_words = post_process_classification(predicted_class, probabilities, important_words)
    
#     # Prepare the output
#     result = f"Predicted class: {final_class}\n\nProbabilities:\n"
#     for label, prob in zip(label_mapping.keys(), probabilities):
#         result += f"{label}: {prob:.4f}\n"
    
#     result += "\nTop 20 Important words:\n"
#     for word, importance in important_words:
#         result += f"{word}: {importance:.4f}\n"
    
#     # Highlight important words in the text
#     highlighted_text = extracted_text
#     for word, _ in important_words:
#         highlighted_text = re.sub(r'\b' + re.escape(word) + r'\b', f'<span style="background-color: yellow;">{word}</span>', highlighted_text, flags=re.IGNORECASE)
    
#     return result, gr.HTML(highlighted_text)

def process_pdf(pdf_file):
    # Extract text from PDF
    extracted_text = extract_text_from_pdf(pdf_file.name)
    
    if not extracted_text:
        return "Failed to extract text from PDF", ""
    
    # Clean the extracted text
    cleaned_text = clean_text(extracted_text)
    
    # Classify the cleaned text in chunks
    predicted_class, avg_probabilities, important_words = classify_text_in_chunks(cleaned_text)
    
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
    gr.Markdown("# Medical Referral Triage Classifier")
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