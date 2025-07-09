"""Text extraction and OCR functionality for PDF processing."""

import tempfile
import os
import time
import logging
import traceback
import pytesseract
from pdf2image import convert_from_path
import re
from config import PatientInfo

async def extract_text_from_pdf(pdf_file):
    """Extract text from PDF using OCR"""
    print(f"Starting text extraction from PDF: {pdf_file if isinstance(pdf_file, str) else pdf_file.name}")
    start_time = time.time()
    
    # Handle both string paths and uploaded files
    if isinstance(pdf_file, str):
        temp_file_path = pdf_file
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf_file.read())
            temp_file_path = temp_file.name

    try:
        # Convert PDF to images
        images = convert_from_path(temp_file_path)
        
        # Extract text from all pages
        full_text = ""
        for i, image in enumerate(images):
            page_text = pytesseract.image_to_string(image)
            full_text += page_text + "\n"
        
        if not full_text.strip():
            logging.warning(f"No text extracted from PDF even after OCR.")
            return "Error: Unable to extract text from PDF. The document might be scanned without OCR or protected."
            
        # Extract the letter content for clinical analysis
        letter_content = extract_letter_content(full_text)
        
        # Return both full text and letter content
        return {
            'full_text': full_text,
            'letter_content': letter_content
        }
        
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {str(e)}")
        logging.error(traceback.format_exc())
        return f"Error: {str(e)}"
    finally:
        if not isinstance(pdf_file, str):
            os.unlink(temp_file_path)
        end_time = time.time()
        print(f"Text extraction completed in {end_time - start_time:.2f} seconds")

def extract_letter_content(text: str) -> str:
    """Extract the main letter content from a referral document"""
    try:
        # Common letter start patterns
        start_patterns = [
            r"(?i)dear (?:dr|doctor|colleague|sir|madam|to whom it may concern)",
            r"(?i)re:|regarding:",
            r"(?i)thank you for (?:referring|seeing|reviewing)",
            r"(?i)i would be grateful",
            r"(?i)i am writing to refer"
        ]
        
        # Common letter end patterns
        end_patterns = [
            r"(?i)yours (?:sincerely|faithfully|truly)",
            r"(?i)kind(?:est)? regards",
            r"(?i)best (?:regards|wishes)",
            r"(?i)many thanks",
            r"(?i)thank you",
            r"(?:Dr|Doctor|Mr|Mrs|Ms|Prof)\.",
            r"(?i)consultant"
        ]
        
        # Find the start of the letter
        letter_start = 0
        for pattern in start_patterns:
            match = re.search(pattern, text)
            if match:
                letter_start = match.start()
                print(f"Found letter start with pattern: {pattern}")
                break
        
        # Find the end of the letter
        letter_end = len(text)
        for pattern in end_patterns:
            matches = list(re.finditer(pattern, text))
            if matches:
                # Take the last match as it's likely the signature
                letter_end = matches[-1].end()
                print(f"Found letter end with pattern: {pattern}")
                break
        
        # Extract the letter content
        letter_content = text[letter_start:letter_end].strip()
        
        # If no clear markers found, look for paragraph structure
        if not letter_content or len(letter_content) < 50:
            paragraphs = text.split('\n\n')
            # Look for paragraphs that look like letter content
            for i, para in enumerate(paragraphs):
                if any(re.search(start, para) for start in start_patterns):
                    letter_content = '\n\n'.join(paragraphs[i:])
                    break
        
        print("\nExtracted Letter Content:")
        print("="*50)
        print(letter_content)
        print("="*50)
        
        if len(letter_content) < 50:
            print("Warning: Extracted content seems too short, falling back to original text")
            return text
            
        return letter_content
        
    except Exception as e:
        logging.error(f"Error extracting letter content: {str(e)}")
        logging.error(traceback.format_exc())
        return text  # Fall back to original text if extraction fails

def extract_patient_info(text: str) -> PatientInfo:
    """Extract patient name and hospital/NHS number from text"""
    result = {
        "name": "Not Found",
        "hospital_number": "Not Found"
    }
    
    # Search a larger portion of the text for names (first 30 lines)
    first_lines = '\n'.join(text.split('\n')[:30])
    search_areas = [first_lines, text]  # Try first lines, then whole text

    # More robust patterns for patient name
    name_patterns = [
        r"(?:Re|RE|re):[\s\n]*([A-Z][A-Za-z-']+(?:\s+[A-Z][A-Za-z-']+){1,3})",
        r"(?:Patient|PATIENT|Name):[\s\n]*([A-Z][A-Za-z-']+(?:\s+[A-Z][A-Za-z-']+){1,3})",
        r"(?:Name|Patient):?\s*([A-Z][A-Za-z-']+(?:\s+[A-Z][A-Za-z-']+){1,3})[\s\n]*(?:DOB|Date of Birth|Born|NHS)",
        r"(?:Dear Dr|Dear Doctor).*?\n.*?(?:Re|RE|re):[\s\n]*([A-Z][A-Za-z-']+(?:\s+[A-Z][A-Za-z-']+){1,3})",
        r"^([A-Z][A-Za-z-']+(?:\s+[A-Z][A-Za-z-']+){1,3})[\s\n]*(?:DOB|Age|Hospital No|NHS No)",
        r"^.*?,\s*([A-Z][A-Za-z-']+(?:\s+[A-Z][A-Za-z-']+){1,3}),.*?(?:DOB|NHS|Hospital)",
        r"(?:regarding|concerning|about):?\s*([A-Z][A-Za-z-']+(?:\s+[A-Z][A-Za-z-']+){1,3})"
    ]
    
    for area in search_areas:
        for pattern in name_patterns:
            match = re.search(pattern, area, re.MULTILINE | re.IGNORECASE)
            if match:
                potential_name = match.group(1).strip().title()
                if not any(title.lower() in potential_name.lower() for title in 
                          ['dr', 'dr.', 'professor', 'prof', 'consultant', 'surgeon', 'gp']):
                    result["name"] = potential_name
                    break
        if result["name"] != "Not Found":
            break

    # NHS/Hospital number patterns (prefer NHS if both found)
    number_patterns = [
        (r"(?:NHS|nhs)[\s\n]*(?:number|no|#)?[\s:]*(\d{3}[\s-]*\d{3}[\s-]*\d{4})", "nhs"),
        (r"(?:Hospital|hosp)[\s\n]*(?:number|no|#)?[\s:]*([A-Z0-9]{6,10})", "hospital"),
        (r"(?:Patient|Registration)[\s\n]*(?:ID|number|no)[\s:]*([A-Z0-9]{6,10})", "hospital"),
        (r"(?:MRN|mrn)[\s:]*([A-Z0-9]{6,10})", "hospital")
    ]
    found_numbers = {}
    for pattern, kind in number_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
            number = match.group(1).strip().upper()
            number = re.sub(r'[\s-]', '', number)
            if kind == "nhs" and re.match(r'^\d{10}$', number):
                found_numbers["nhs"] = number
            elif kind == "hospital" and re.match(r'^[A-Z0-9]{6,10}$', number):
                found_numbers.setdefault("hospital", number)
    if "nhs" in found_numbers:
        result["hospital_number"] = found_numbers["nhs"]
    elif "hospital" in found_numbers:
        result["hospital_number"] = found_numbers["hospital"]

    return result

def extract_xray_sentences(text: str) -> str:
    """Extract sentences mentioning x-ray/imaging for LLM analysis"""
    xray_keywords = [
        "x-ray", "xray", "radiograph", "imaging", "scan", "mri", "ct", "xrays", "radiology", "ultrasound"
    ]
    sentences = re.split(r'(?<=[\.\n])\s*', text)
    xray_sentences = [
        s.strip() for s in sentences
        if any(kw in s.lower() for kw in xray_keywords)
    ]
    return " ".join(xray_sentences) if xray_sentences else ""