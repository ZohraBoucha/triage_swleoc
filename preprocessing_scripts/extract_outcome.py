import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm
import csv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

model_id = "HPAI-BSC/Llama3-Aloe-8B-Alpha"  # Changed to a publicly available model

# Load the LLaMA model and tokenizer
def load_model_and_tokenizer():
    try:
        logging.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logging.info("Tokenizer loaded successfully.")
        
        logging.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        logging.info("Model loaded successfully.")
        
        # Create a new GenerationConfig
        generation_config = GenerationConfig.from_model_config(model.config)
        
        # Ensure pad_token_id is set to an integer
        if isinstance(model.config.pad_token_id, list):
            model.config.pad_token_id = model.config.pad_token_id[0]
        elif model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id
        
        generation_config.pad_token_id = model.config.pad_token_id
        model.generation_config = generation_config
        
        logging.info(f"pad_token_id set to: {model.generation_config.pad_token_id}")
        
        return model, tokenizer
    except Exception as e:
        logging.error(f"Error in load_model_and_tokenizer: {str(e)}", exc_info=True)
        raise

# Function to extract outcome using LLaMA model
def extract_outcome(model, tokenizer, text):
    if not isinstance(text, str) or pd.isna(text):
        logging.warning(f"Invalid or missing clinic text: {text}")
        return "Invalid or missing clinic text"
    
    prompt = f"""Review the provided medical clinic letter and determine the appropriate category based on the presence of surgical recommendations related to the following conditions:

'Total Hip'
'Revision Hip'
'Soft Tissue Hip'
'Unicompartmental'
'Knee Revision'
'Injection'
'Soft Tissue Knee'
Classification Categories:

'Needs Surgery': Assign this category if the letter explicitly mentions that surgery is required or recommended for any of the listed conditions.
'Needs Revision': Choose this if the letter suggests a need for revision surgery or a significant adjustment to a previous treatment plan specifically for the conditions listed.
'More details required': Use this category if the letter does not clearly indicate a need for surgery or a revision, or if it lacks sufficient details to determine the necessity of these procedures.
Important Note: Focus only on potential upcoming surgeries as indicated in the consultation. Do not consider mentions of past surgeries. Choose 'More details required' if there’s ambiguity about the need for upcoming surgery or revision.

Text of Medical Clinic Letter: {text}

Based on the letter above, the outcome is (choose only one category):

"""
    
    try:
        logging.info("Tokenizing input...")
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        logging.info(f"Input shape: {inputs.input_ids.shape}")
        
        logging.info("Moving input to device...")
        inputs = inputs.to(model.device)
        
        logging.info("Generating output...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                num_return_sequences=1,
                pad_token_id=model.generation_config.pad_token_id,
                temperature=0.7,  # Added temperature for some variability
                top_p=0.95,  # Added top_p for nucleus sampling
            )
        
        logging.info("Decoding output...")
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.info(f"Raw response: {response}")
        
        # Extract the outcome from the response
        outcome = response.split("Based on the above letter, the outcome is")[-1].strip().lower()
        logging.info(f"Extracted outcome: {outcome}")
        
        if "needs surgery" in outcome:
            return "Needs Surgery"
        elif "needs revision" in outcome:
            return "Needs Revision"
        else:
            return "More details required"
    except Exception as e:
        logging.error(f"Error in extract_outcome: {str(e)}", exc_info=True)
        return "Error in processing"

# Main function to process the CSV file
def process_csv(input_file, output_file, num_patients=10):
    model, tokenizer = load_model_and_tokenizer()
    
    df = pd.read_csv(input_file)
    
    # Limit to the first num_patients
    df = df.head(num_patients)
    
    outcomes = []
    
    # Create a progress bar
    pbar = tqdm(total=len(df), desc="Processing patients")
    
    logging.info(f"\nProcessing first {num_patients} patients:\n")
    
    for index, row in df.iterrows():
        hospital_number = row['Hospital Number']
        clinic_text = row['Clinic (C)']
        
        # Update progress bar description with current patient
        pbar.set_description(f"Processing patient {hospital_number}")
        
        outcome = extract_outcome(model, tokenizer, clinic_text)
        outcomes.append(outcome)
        
        # Log information for each patient
        logging.info(f"Patient {hospital_number}:")
        if isinstance(clinic_text, str):
            logging.info(f"Clinic Text: {clinic_text[:100]}...")  # Log first 100 characters of clinic text
        else:
            logging.info("Clinic Text: Invalid or missing")
        logging.info(f"Extracted Outcome: {outcome}")
        logging.info("-" * 50)
        
        # Update progress bar
        pbar.update(1)
    
    # Close progress bar
    pbar.close()
    
    df['Outcome'] = outcomes
    df.to_csv(output_file, index=False)
    logging.info(f"\nResults for first {num_patients} patients saved to {output_file}")


if __name__ == "__main__":
    input_file = "extracted_letters_text.csv"
    output_file = "extracted_outcome.csv"
    process_csv(input_file, output_file)