import os
import pandas as pd
import re
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

def extract_text_from_pdf(pdf_path):
    print(f"Extracting text from: {pdf_path}")
    try:
        doc = DocumentFile.from_pdf(pdf_path)
        predictor = ocr_predictor(pretrained=True)
        result = predictor(doc)
        json_output = result.export()
        
        extracted_text = ""
        for page in json_output['pages']:
            for block in page['blocks']:
                for line in block['lines']:
                    for word in line['words']:
                        extracted_text += word['value'] + ' '
                    extracted_text += '\n'
        
        print(f"Successfully extracted text from: {pdf_path}")
        return extracted_text.strip()
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {str(e)}")
        return ""

def categorize_pdf(filename):
    if re.search(r'c', filename, re.IGNORECASE):
        return 'C'
    else:
        return 'R'  # Default to 'R' if not explicitly 'C'

def process_hospital_folders(root_dir):
    print(f"Starting to process directory: {root_dir}")
    data = []
    folder_count = 0

    if not os.path.exists(root_dir):
        print(f"Error: The directory {root_dir} does not exist.")
        return pd.DataFrame()

    all_items = os.listdir(root_dir)
    folders = [item for item in all_items if os.path.isdir(os.path.join(root_dir, item))]
    total_folders = len(folders)
    print(f"Total folders found: {total_folders}")

    for hospital_folder in folders:
        folder_count += 1
        hospital_path = os.path.join(root_dir, hospital_folder)
        print(f"\nProcessing hospital folder {folder_count}/{total_folders}: {hospital_folder}")
        
        c_texts = []
        r_texts = []

        pdf_files = [f for f in os.listdir(hospital_path) if f.lower().endswith('.pdf')]
        print(f"PDF files found: {pdf_files}")

        for filename in pdf_files:
            full_path = os.path.join(hospital_path, filename)
            category = categorize_pdf(filename)
            extracted_text = extract_text_from_pdf(full_path)
            
            if category == 'C':
                print(f"Processed 'C' file: {filename}")
                c_texts.append(extracted_text)
            else:
                print(f"Processed 'R' file: {filename}")
                r_texts.append(extracted_text)

        data.append({
            'Hospital Number': hospital_folder,
            'Referrals (R)': '\n\n'.join(r_texts).strip(),
            'Clinic (C)': '\n\n'.join(c_texts).strip()
        })
        print(f"Added data for hospital number: {hospital_folder}")
        print(f"Number of 'R' files combined: {len(r_texts)}")
        print(f"Number of 'C' files combined: {len(c_texts)}")

        # Print progress
        if folder_count % 10 == 0:
            print(f"Processed {folder_count}/{total_folders} folders")

    print(f"\nFinished processing.")
    print(f"Total folders processed: {folder_count}")
    print(f"Total entries in dataset: {len(data)}")
    return pd.DataFrame(data)

# Example usage
root_directory = '/home/swleocresearch/Desktop/triage-ai/datasets/cleaned_and_sorted'
print(f"Starting script with root directory: {root_directory}")

try:
    df = process_hospital_folders(root_directory)

    # Save the results to a CSV file
    output_file = 'extracted_letters_text.csv'
    df.to_csv(output_file, index=False)
    print(f"Text extraction complete. Results saved to '{output_file}'")
    print(f"Total rows in the CSV: {len(df)}")
    print(f"Columns in the CSV: {', '.join(df.columns)}")
except Exception as e:
    print(f"An error occurred during script execution: {str(e)}")

print("Script execution finished.")