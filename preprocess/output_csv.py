import os
import csv
import pandas as pd
import re
import matplotlib.pyplot as plt
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

def extract_text_from_pdf(pdf_path):
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
    
    return extracted_text.strip()

def extract_patient_info_from_filename(filename):
    # Use regex to extract name and NHS number from filename
    match = re.match(r'(.*?)\s*\((\d+)\)\.pdf', filename)
    if match:
        return match.group(1).strip(), match.group(2)
    return None, None

def main():
    # Read outcomes from Excel file
    outcomes_df = pd.read_excel('datasets/referrals/Hip eRS referral outcomes.xlsx')
    
    # Create a dictionary to store outcomes
    # Assuming the Excel file has 'NHS_Number' and 'Outcome' columns
    outcomes_dict = dict(zip(outcomes_df['NHS Number'], outcomes_df['HK MDT TRIAGE COMMENT']))
    
    # Prepare data for output.csv
    output_data = []
    
    # Process PDF files
    pdf_folder = 'datasets/referrals'
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            input_text = extract_text_from_pdf(pdf_path)
            
            # Extract patient name and NHS number from the filename
            patient_name, nhs_number = extract_patient_info_from_filename(pdf_file)
            
            if patient_name is None or nhs_number is None:
                print(f"Warning: Could not extract patient info from filename: {pdf_file}")
                continue
            
            # Get outcome from the outcomes dictionary
            outcome = outcomes_dict.get(nhs_number, "No outcome found")
            
            # Append data to output_data list
            output_data.append({
                'Patient Name': patient_name,
                'NHS_Number': nhs_number,
                'Input': input_text,
                'Outcome': outcome,
                'Instruction': 'Predict the outcome based on the referral letter information.'
            })
    
    # Write output to CSV
    with open('output.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Patient Name', 'NHS_Number', 'Input', 'Outcome', 'Instruction']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in output_data:
            writer.writerow(row)
    
    print("output.csv has been created successfully.")

if __name__ == "__main__":
    main()
