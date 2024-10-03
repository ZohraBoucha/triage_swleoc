import torch
import transformers
import time
import pandas as pd
from tqdm import tqdm

print("Step 1: Starting script...")

print("Step 2: Checking CUDA availability...")
if torch.cuda.is_available():
    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Using CPU. This may be very slow.")

print("Step 3: Preparing to load the model...")
model_id = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"

print("Step 4: Loading the CSV file...")
df = pd.read_csv('/home/swleocresearch/Desktop/triage-ai/datasets/triage_dataset.csv', encoding='latin-1')
df = df[df['output'].isin(['surgery', 'discharge'])]
print(f"Loaded {len(df)} referral letters.")

print("Step 5: Loading the model (this may take a while)...")
model_start_time = time.time()
try:
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={
            "torch_dtype": torch.float16,
        },
        device_map="auto",
    )
    model_end_time = time.time()
    print(f"Model loaded successfully in {model_end_time - model_start_time:.2f} seconds.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

print("Step 6: Processing the referral letters...")
start_time = time.time()

def process_referral(referral_text):
    prompt = f"""Remove the following for this referral letter:
    - Administrative details
    - Headings
    - Addresses
    - Phone numbers
    
    Original referral letter:
    {referral_text}

    Extracted medical information:
    """
    
    response = pipeline(prompt, max_new_tokens=500, do_sample=True, temperature=0.7, num_return_sequences=1)[0]['generated_text']
    
    # Extract only the generated part
    extracted_text = response.split("Extracted medical information:")[-1].strip()
    
    return extracted_text

try:
    tqdm.pandas(desc="Processing referrals")
    df['processed_referral'] = df['referral'].progress_apply(process_referral)
    
    end_time = time.time()
    processing_time = end_time - start_time

    print(f"\nProcessing completed in {processing_time:.2f} seconds.")

    print("\nStep 7: Saving results...")
    output_file = 'processed_referrals.csv'
    df[['processed_referral', 'output']].to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

except Exception as e:
    print(f"Error during processing: {e}")

print("Script finished.")