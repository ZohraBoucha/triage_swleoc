import sys
import os

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import the patch before any other imports
import torch
if not hasattr(torch.library, 'register_fake'):
    def register_fake(target):
        def decorator(fn):
            return fn
        return decorator
    torch.library.register_fake = register_fake
    print("🔧 Applied torch.library.register_fake patch")

# Now import the rest of your modules
try:
    from inference.consultant_triage import ConsultantManager, display_consultant_management
    from consultant_triage import ConsultantManager as ConsultantManager2, display_consultant_management as display_consultant_management2
except ImportError:
    # Mock the consultant management if not available
    class ConsultantManager:
        def find_suitable_consultants(self, *args, **kwargs): return []
        def get_next_available_slot(self, *args, **kwargs): return None
    def display_consultant_management(): 
        import streamlit as st
        st.header("Consultant Management (Module not found)")

import tempfile
import base64
import asyncio
import logging
import time
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from langchain_ollama import OllamaLLM
import json
import traceback
import pytesseract
from pdf2image import convert_from_path
import re
import uuid
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Union, TypedDict, Optional
from functools import partial
import streamlit as st
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)

# Initialize the OCR model
try:
    model = ocr_predictor(pretrained=True)
except Exception as e:
    logging.warning(f"Could not initialize OCR model: {e}")
    model = None

# Initialize the Llama model
try:
    llm = OllamaLLM(model="llama3.3:70b")
except Exception as e:
    logging.warning(f"Could not initialize LLM: {e}")
    # Fallback to mock LLM
    class MockLLM:
        def invoke(self, prompt: str) -> str:
            mock_response = {
                "procedure_type": "arthroplasty",
                "body_part": "knee",
                "arthroplasty_type": "primary",
                "further_information_needed": "no",
                "had_injections": "yes",
                "had_physiotherapy": "no",
                "confidence": 0.92
            }
            return json.dumps(mock_response)
    llm = MockLLM()

# Valid values for medical referral fields
VALID_PROCEDURE_TYPES = ['arthroplasty', 'soft_tissue', 'unknown']
VALID_BODY_PARTS = ['hip', 'knee', 'unknown']
VALID_ARTHROPLASTY_TYPES = ['primary', 'revision', 'na', 'unknown']
VALID_INJECTIONS = ['yes', 'no']
VALID_PHYSIOTHERAPY = ['yes', 'no']
VALID_FURTHER_INFO = ['yes', 'no']

class PatientInfo(TypedDict):
    name: str
    hospital_number: str

class MedicalReferral(BaseModel):
    procedure_type: str = Field(description="The type of procedure (arthroplasty, soft tissue)")
    body_part: str = Field(description="The body part involved (hip, knee)")
    arthroplasty_type: str = Field(description="The type of arthroplasty (primary, revision)")
    further_information_needed: str = Field(description="Whether further information is needed (yes, no)")
    had_injections: str = Field(description="Whether the patient had injections (yes, no, unknown)")
    had_physiotherapy: str = Field(description="Whether the patient had physiotherapy (yes, no, unknown)")

class AnalysisResult(TypedDict):
    procedure_type: str
    body_part: str
    arthroplasty_type: str
    further_information_needed: str
    had_injections: str
    had_physiotherapy: str
    confidence: float
    name: str
    hospital_number: str
    xray_findings: str

async def extract_text_from_pdf(pdf_file):
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

def summarize_xray_findings_with_llm(xray_text: str) -> str:
    if not xray_text.strip():
        return "No X-ray findings reported"
    
    prompt = (
        "You are a medical scribe extracting X-ray or imaging findings from referral letters. "
        "Your task is to extract EXACTLY what the X-ray or imaging showed, including the SPECIFIC body part mentioned. "
        "IMPORTANT: If the text explicitly mentions a body part with the findings (like 'hip', 'knee', 'shoulder', etc.), "
        "you MUST include that exact body part in your summary. Do not replace specific body parts with placeholder text. "
        "\n\nExamples:"
        "\nInput: 'We reviewed the recent x-ray reported as moderate degenerative change through the left hip.'"
        "\nCorrect output: 'Moderate degenerative change in the left hip.'"
        "\nIncorrect output: 'Moderate degenerative change in left [unspecified body part].'"
        "\n\nInput: 'X-rays show severe osteoarthritis of the right knee with joint space narrowing.'"
        "\nCorrect output: 'Severe osteoarthritis of the right knee with joint space narrowing.'"
        "\n\nYour summary should be concise (under 20 words) and contain ONLY what was explicitly mentioned in the imaging report."
        "\n\nText with X-ray findings:\n{xray_text}\n\n"
        "X-ray findings summary:"
    )

    prompt = prompt.format(xray_text=xray_text)
    
    try:
        summary = llm.invoke(prompt)
        return summary.strip()
    except Exception as e:
        logging.error(f"LLM error for x-ray findings: {str(e)}")
        return "Error extracting X-ray findings"

def analyze_medical_referral(text) -> AnalysisResult:
    """Rule-based analysis for fallback"""
    try:
        result = {
            "procedure_type": "unknown",
            "body_part": "unknown",
            "arthroplasty_type": "unknown",
            "further_information_needed": "yes",
            "had_injections": "no",
            "had_physiotherapy": "no",
            "confidence": 0.0,
            "name": "Not Found",
            "hospital_number": "Not Found",
            "xray_findings": "No"
        }
        
        # Extract patient info
        patient_info = extract_patient_info(text)
        result.update(patient_info)
        
        # Extract X-ray findings
        xray_text = extract_xray_sentences(text)
        result["xray_findings"] = summarize_xray_findings_with_llm(xray_text)
        
        text_lower = text.lower()

        # Body part detection
        if re.search(r'hip', text_lower):
            result["body_part"] = "hip"
        elif re.search(r'knee', text_lower):
            result["body_part"] = "knee"

        # Procedure type
        if re.search(r'(arthroplasty|replacement|tkr|thr|joint replacement)', text_lower):
            result["procedure_type"] = "arthroplasty"
            result["arthroplasty_type"] = "primary"
            result["confidence"] = 0.8
            result["further_information_needed"] = 'no'
            if re.search(r'revision|previous.*replacement|failed.*implant|loosening', text_lower):
                result["arthroplasty_type"] = "revision"
        else:
            result["procedure_type"] = "soft_tissue"
            result["arthroplasty_type"] = "na"

        # Injections
        if re.search(r'(injection|injections|steroid|cortisone)', text_lower):
            result["had_injections"] = "yes"
        
        # Physiotherapy
        if re.search(r'(physio|physiotherapy|physical therapy)', text_lower):
            result["had_physiotherapy"] = "yes"
            
        return result
    except Exception as e:
        logging.error(f"Error in analyze_medical_referral: {str(e)}")
        return {
            "procedure_type": "unknown",
            "body_part": "unknown",
            "arthroplasty_type": "unknown",
            "further_information_needed": "yes",
            "had_injections": "no",
            "had_physiotherapy": "no",
            "confidence": 0.0,
            "name": "Not Found",
            "hospital_number": "Not Found",
            "xray_findings": "No"
        }

def validate_analysis_result(result):
    """Validate the analysis result against valid categories"""
    try:
        # Check all required fields are present
        required_fields = ['procedure_type', 'body_part', 'arthroplasty_type', 'further_information_needed', 'confidence', 'had_injections', 'had_physiotherapy']
        if not all(field in result for field in required_fields):
            missing = [f for f in required_fields if f not in result]
            return False, f"Missing required fields: {missing}"
            
        # Validate against allowed values
        if result['procedure_type'].lower() not in VALID_PROCEDURE_TYPES: 
            return False, f"Invalid procedure_type: {result['procedure_type']}"
            
        if result['body_part'].lower() not in VALID_BODY_PARTS:
            return False, f"Invalid body_part: {result['body_part']}"
            
        if result['arthroplasty_type'].lower() not in VALID_ARTHROPLASTY_TYPES:
            return False, f"Invalid arthroplasty_type: {result['arthroplasty_type']}"
            
        if result['further_information_needed'].lower() not in VALID_FURTHER_INFO:
            return False, f"Invalid further_information_needed: {result['further_information_needed']}"
            
        if result['had_injections'].lower() not in VALID_INJECTIONS:
            return False, f"Invalid had_injections: {result['had_injections']}"
            
        if result['had_physiotherapy'].lower() not in VALID_PHYSIOTHERAPY:
            return False, f"Invalid had_physiotherapy: {result['had_physiotherapy']}"
            
        # Check if too many unknowns
        unknown_count = sum(1 for value in result.values() if str(value).lower() == 'unknown')
        if unknown_count >= 2:
            return False, "Too many unknown values"
            
        return True, ""
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def analyze_medical_referral_with_llama(text):
    try:
        # Extract patient info first
        patient_info = extract_patient_info(text)
        
        # Add example to prompt to guide the model
        prompt = f"""You are an orthopedic referral analysis assistant. Analyse this referral letter carefully:

TEXT TO ANALYSE:
{text}

First, explain your reasoning step by step. Then provide your classification in the exact JSON format shown below.
You must choose from these specific values:
- procedure_type: {VALID_PROCEDURE_TYPES}
- body_part: {VALID_BODY_PARTS}
- arthroplasty_type: {VALID_ARTHROPLASTY_TYPES}
- further_information_needed: {VALID_FURTHER_INFO} 
- had_injections: ["yes", "no"]
- had_physiotherapy: ["yes", "no"]

Analysis steps:
1. Identify the main joint/location of complaint
2. Assess severity and impact on life
3. Look for imaging findings
4. Consider current treatments - carefully check for ANY mention of injections or physiotherapy
5. Check for any previous surgery history
6. Determine if surgical intervention is indicated

Important: For had_injections and had_physiotherapy, default to "no" if there is no explicit mention in the text.

After your analysis, provide your classification in this EXACT format:
{{
    "procedure_type": "arthroplasty",
    "body_part": "hip",
    "arthroplasty_type": "primary",
    "further_information_needed": "no",
    "confidence": 0.9,
    "had_injections": "no",
    "had_physiotherapy": "no"
}}

YOUR ANALYSIS AND RESPONSE:"""

        # Get LLM's analysis
        print("\n" + "="*50)
        print("SENDING TO LLM:")
        print(f"Text length: {len(text)} characters")
        print("="*50)
        
        llm_response = llm.invoke(prompt)
        
        print("\n" + "="*50)
        print("LLM FULL RESPONSE:")
        print(llm_response)
        print("="*50)
        
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'\{[\s\S]*\}', llm_response)
            if json_match:
                json_str = json_match.group()
                print("\nExtracted JSON:")
                print(json_str)
                
                # Parse LLM's JSON response
                llm_result = json.loads(json_str)
                
                # Convert all string values to lowercase for consistency
                llm_result = {k: v.lower() if isinstance(v, str) else v for k, v in llm_result.items()}
                
                # Validate the result
                is_valid, error_msg = validate_analysis_result(llm_result)
                if is_valid:
                    print("\nVALID ANALYSIS RESULT:")
                    print(json.dumps(llm_result, indent=2))
                    
                    # Add patient info to the result
                    llm_result.update(patient_info)
                    
                    # Extract X-ray findings
                    xray_text = extract_xray_sentences(text)
                    llm_result['xray_findings'] = summarize_xray_findings_with_llm(xray_text)
                    
                    return llm_result
                    
                print(f"\nINVALID ANALYSIS: {error_msg}")
                
                # Only fall back to rule-based if core fields are unknown
                core_fields_unknown = (
                    llm_result.get('procedure_type', 'unknown') == 'unknown' or
                    llm_result.get('body_part', 'unknown') == 'unknown' or
                    (llm_result.get('procedure_type') == 'arthroplasty' and 
                     llm_result.get('arthroplasty_type', 'unknown') == 'unknown')
                )
                
                if core_fields_unknown:
                    print("Core fields unknown - falling back to rule-based analysis...")
                    return analyze_medical_referral(text)
                else:
                    print("Using LLM analysis despite validation warning")
                    # Add patient info and X-ray findings
                    llm_result.update(patient_info)
                    xray_text = extract_xray_sentences(text)
                    llm_result['xray_findings'] = summarize_xray_findings_with_llm(xray_text)
                    return llm_result
            
            else:
                print("\nNo JSON found in LLM response")
                print("Falling back to rule-based analysis...")
            
            # Get rule-based analysis
            rule_result = analyze_medical_referral(text)
            print("\nRULE-BASED ANALYSIS RESULT:")
            print(json.dumps(rule_result, indent=2))
            return rule_result
            
        except json.JSONDecodeError as e:
            print(f"\nJSON PARSE ERROR: {str(e)}")
            print("Falling back to rule-based analysis...")
            return analyze_medical_referral(text)
            
    except Exception as e:
        logging.error(f"Error in analyze_medical_referral_with_llama: {str(e)}")
        logging.error(traceback.format_exc())
        return {
            "procedure_type": "unknown",
            "body_part": "unknown",
            "arthroplasty_type": "unknown",
            "further_information_needed": "yes",
            "had_injections": "no",
            "had_physiotherapy": "no",
            "confidence": 0.0,
            "name": "Not Found",
            "hospital_number": "Not Found",
            "xray_findings": "No"
        }

# Data management functions
def save_analysis_changes(original_result, modified_result, analysis_id, filename):
    """Save all field values, marking which ones were changed"""
    valid_fields = ['procedure_type', 'body_part', 'arthroplasty_type', 'further_information_needed', 'had_injections', 'had_physiotherapy']
    
    # Check if any changes were made
    changes_made = False
    for key in valid_fields:
        orig_val = str(original_result.get(key, '')).lower()
        mod_val = str(modified_result.get(key, '')).lower()
        
        if orig_val != mod_val:
            changes_made = True
            break
    
    if changes_made:
        data = []
        for key in valid_fields:
            data.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'analysis_id': analysis_id,
                'filename': filename,
                'field': key,
                'original_value': str(original_result.get(key, '')).lower(),
                'modified_value': str(modified_result.get(key, '')).lower(),
                'was_changed': str(original_result.get(key, '')).lower() != str(modified_result.get(key, '')).lower()
            })
        
        df = pd.DataFrame(data)
        csv_path = 'datasets/analysis_changes.csv'
        
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)
        
        return True
    return False

def save_analysis_history(result, analysis_id, filename, model_type):
    """Save original analysis results"""
    try:
        os.makedirs('datasets', exist_ok=True)
        df = pd.DataFrame([result])
        df['timestamp'] = datetime.now().isoformat()
        df['analysis_id'] = analysis_id
        df['filename'] = filename
        df['model_type'] = model_type
        path = 'datasets/analysis_history.csv'
        df.to_csv(path, mode='a', header=not os.path.exists(path), index=False)
        return True
    except Exception as e:
        logging.error(f"Save history error: {e}")
        return False

def ensure_data_directories():
    """Create necessary directories"""
    os.makedirs('datasets', exist_ok=True)

@st.cache_data
def get_image_as_base64(file_path):
    """Get image as base64 string"""
    try:
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return None

@st.cache_data
def cached_analysis(text, analysis_id):
    """Cached analysis function"""
    try:
        result = analyze_medical_referral_with_llama(text)
        # Ensure all required fields are present
        default_fields = {
            "procedure_type": "unknown",
            "body_part": "unknown",
            "arthroplasty_type": "unknown",
            "further_information_needed": "yes",
            "had_injections": "no",
            "had_physiotherapy": "no",
            "confidence": 0.0,
            "name": "Not Found",
            "hospital_number": "Not Found",
            "xray_findings": "No"
        }
        # Update default fields with any values from the analysis
        default_fields.update(result)
        return default_fields
    except Exception as e:
        logging.error(f"Error in cached analysis: {str(e)}")
        return {
            "procedure_type": "unknown",
            "body_part": "unknown",
            "arthroplasty_type": "unknown",
            "further_information_needed": "yes",
            "had_injections": "no",
            "had_physiotherapy": "no",
            "confidence": 0.0,
            "name": "Not Found",
            "hospital_number": "Not Found",
            "xray_findings": "No"
        }

def determine_priority(result):
    """Determine priority based on analysis result"""
    if result.get('arthroplasty_type') == 'revision':
        return 'Immediate'
    if result.get('procedure_type') == 'arthroplasty':
        return 'Urgent'
    return 'Routine'

# Main Streamlit UI
def main():
    ensure_data_directories()
    st.set_page_config(page_title="NHS Referral Management", layout="wide")
    
    # CSS Styles - Keep the existing NHS-styled CSS
    st.markdown("""<style>
    :root { --nhs-blue: #005EB8; --nhs-dark-blue: #003087; --nhs-text: #212B32; --nhs-bg: #F0F4F5; --nhs-container-bg: #FFFFFF; --nhs-border: #E8EDEE; --nhs-red: #DA291C; --nhs-orange: #ED8B00; --nhs-green: #007F3B;}
    .stApp { background-color: var(--nhs-bg); color: var(--nhs-text); }
    #MainMenu, footer, header { visibility: hidden; }
    .app-header { display: flex; align-items: center; justify-content: space-between; padding: 1rem 2rem; background-color: var(--nhs-container-bg); border-bottom: 4px solid var(--nhs-blue); margin: -3rem -3rem 1.5rem -3rem; }
    .header-title { color: var(--nhs-text); font-weight: 600; font-size: 1.75rem; margin: 0; }
    .header-logo { max-height: 80px; }
    .stTabs [aria-selected="true"] { border-bottom: 4px solid var(--nhs-blue); }
    h2, h3 { color: var(--nhs-dark-blue); border-bottom: 1px solid var(--nhs-border); padding-bottom: 0.5rem; }
    .referral-card { border: 1px solid var(--nhs-border); border-left-width: 8px; border-radius: 4px; padding: 1.5rem; margin-bottom: 1.5rem; background-color: var(--nhs-container-bg); }
    .priority-Immediate { border-left-color: var(--nhs-red); }
    .priority-Urgent { border-left-color: var(--nhs-orange); }
    .priority-Routine { border-left-color: var(--nhs-blue); }
    .card-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1rem; }
    .patient-name { font-size: 1.2rem; font-weight: bold; }
    .patient-id { font-size: 0.9rem; color: #555; }
    .priority-tag { font-size: 0.8rem; font-weight: bold; padding: 0.2rem 0.6rem; border-radius: 4px; color: var(--nhs-text); }
    .tag-Immediate { background-color: #FEE; border: 1px solid var(--nhs-red); }
    .tag-Urgent { background-color: #FEF3D7; border: 1px solid var(--nhs-orange); }
    .tag-Routine { background-color: #EBF4FF; border: 1px solid var(--nhs-blue); }
    .card-body { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
    .info-group { background-color: var(--nhs-bg); padding: 1rem; border-radius: 4px; }
    .info-group h4 { font-size: 0.9rem; color: var(--nhs-dark-blue); margin-top:0; margin-bottom: 0.75rem; border-bottom: 1px solid var(--nhs-border); padding-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.5px;}
    .info-item { display: flex; justify-content: space-between; font-size: 0.9rem; margin-bottom: 0.5rem; }
    .info-label { color: #555; }
    .info-value { font-weight: bold; }
    .findings-box { grid-column: 1 / -1; margin-top: 1rem; background-color: var(--nhs-bg); padding: 1rem; border-radius: 4px; }
    .card-actions { margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid var(--nhs-border); display: flex; gap: 0.75rem; justify-content: flex-end; }
    </style>""", unsafe_allow_html=True)
    
    # Logo and header
    logo_path = "assets/swleoc_logo.png"
    if logo_base64 := get_image_as_base64(logo_path):
        st.markdown(f'<div class="app-header"><img src="data:image/png;base64,{logo_base64}" class="header-logo"><h1 class="header-title">Referral Management</h1></div>', unsafe_allow_html=True)
    else:
        st.title("🏥 Referral Management")

    st.cache_data.clear()
    tab1, tab2, tab3 = st.tabs(["Active Referrals", "Analytics Dashboard", "System Configuration"])
    
    with tab1:
        handle_referrals_tab()
    with tab2:
        st.header("Analytics Placeholder")
    with tab3:
        display_consultant_management()

def handle_referrals_tab():
    """Handle the referrals tab with upload and analysis"""
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Upload Referrals")
        uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type="pdf", label_visibility="collapsed")
        if uploaded_files:
            if st.button("🚀 Analyze All", use_container_width=True, type="primary"):
                process_all_files(uploaded_files)
    
    with col2:
        st.subheader("Triage Worklist")
        display_referral_queue(uploaded_files if 'uploaded_files' in locals() else [])

def process_all_files(uploaded_files):
    """Process all uploaded files with real OCR and analysis"""
    for uploaded_file in uploaded_files:
        file_key = f"analyzed_{uploaded_file.name}"
        if file_key not in st.session_state:
            with st.spinner(f"Analyzing {uploaded_file.name}..."):
                try:
                    # Extract text from PDF
                    extracted_text = asyncio.run(extract_text_from_pdf(uploaded_file))
                    
                    if isinstance(extracted_text, str) and extracted_text.startswith("Error:"):
                        st.error(f"Error processing {uploaded_file.name}: {extracted_text}")
                        continue
                    
                    analysis_id = str(uuid.uuid4())[:8]
                    
                    # Run analysis on the extracted text
                    if isinstance(extracted_text, dict):
                        result = cached_analysis(extracted_text['letter_content'], analysis_id)
                        # Update with additional info from full text
                        patient_info = extract_patient_info(extracted_text['full_text'])
                        result.update(patient_info)
                        
                        # Extract X-ray findings
                        xray_text = extract_xray_sentences(extracted_text['full_text'])
                        result['xray_findings'] = summarize_xray_findings_with_llm(xray_text)
                    else:
                        # Fallback if extraction format is different
                        result = cached_analysis(extracted_text, analysis_id)
                    
                    st.session_state[file_key] = {'analysis_id': analysis_id, 'result': result}
                    
                    # Save to history
                    save_analysis_history(result, analysis_id, uploaded_file.name, "llm_analysis")
                    
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                    logging.error(traceback.format_exc())
    
    st.success("Analysis complete!")
    time.sleep(1)
    st.rerun()

def display_referral_queue(uploaded_files):
    """Display the referral queue with analyzed files"""
    if not uploaded_files:
        st.info("Upload files to see the worklist.")
        return
    
    referrals_to_display = [f for f in uploaded_files if f"analyzed_{f.name}" in st.session_state]
    if not referrals_to_display:
        st.info("Click 'Analyze All' to process uploaded files.")
        return
    
    for f in referrals_to_display:
        display_referral_card(f, st.session_state[f"analyzed_{f.name}"])

def display_referral_card(uploaded_file, data):
    """Display a referral card with NHS styling"""
    result, analysis_id = data['result'], data['analysis_id']
    priority = determine_priority(result)
    
    with st.container(border=True, key=f"card_{analysis_id}"):
        # Card Header
        st.markdown(f"""<div class="card-header">
            <div><div class="patient-name">{result.get('name', 'N/A')}</div><div class="patient-id">NHS: {result.get('hospital_number', 'N/A')}</div></div>
            <div class="priority-tag tag-{priority}">{priority.upper()}</div>
        </div>""", unsafe_allow_html=True)

        # Card Body with Groups
        st.markdown('<div class="card-body">', unsafe_allow_html=True)
        
        # Clinical Classification Group
        st.markdown('<div class="info-group"><h4>Clinical Classification</h4>', unsafe_allow_html=True)
        st.markdown(f"""<div class="info-item"><span class="info-label">Procedure</span><span class="info-value">{result.get('procedure_type','').replace('_',' ').title()}</span></div>
                        <div class="info-item"><span class="info-label">Body Part</span><span class="info-value">{result.get('body_part','').title()}</span></div>
                        <div class="info-item"><span class="info-label">Type</span><span class="info-value">{result.get('arthroplasty_type','').title()}</span></div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # History & Flags Group
        st.markdown('<div class="info-group"><h4>History & Flags</h4>', unsafe_allow_html=True)
        st.markdown(f"""<div class="info-item"><span class="info-label">Injections</span><span class="info-value">{result.get('had_injections','').title()}</span></div>
                        <div class="info-item"><span class="info-label">Physiotherapy</span><span class="info-value">{result.get('had_physiotherapy','').title()}</span></div>
                        <div class="info-item"><span class="info-label">Info Needed</span><span class="info-value">{result.get('further_information_needed','').title()}</span></div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Imaging Findings Box
        if xray := result.get('xray_findings'):
            st.markdown(f'<div class="findings-box"><h4>Imaging Findings</h4><p>{xray}</p></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Modification Expander
        if st.session_state.get(f'modify_{analysis_id}', False):
            with st.form(key=f"form_{analysis_id}"):
                st.markdown("**Correct the AI's classification:**")
                form_col1, form_col2 = st.columns(2)
                
                with form_col1:
                    proc_type = st.selectbox('Procedure Type', VALID_PROCEDURE_TYPES, index=VALID_PROCEDURE_TYPES.index(result.get('procedure_type', 'unknown')))
                    body_part = st.selectbox('Body Part', VALID_BODY_PARTS, index=VALID_BODY_PARTS.index(result.get('body_part', 'unknown')))
                    arthro_type = st.selectbox('Arthroplasty Type', VALID_ARTHROPLASTY_TYPES, index=VALID_ARTHROPLASTY_TYPES.index(result.get('arthroplasty_type', 'unknown')))
                
                with form_col2:
                    injections = st.selectbox('Had Injections', VALID_INJECTIONS, index=VALID_INJECTIONS.index(result.get('had_injections', 'no')))
                    physio = st.selectbox('Had Physiotherapy', VALID_PHYSIOTHERAPY, index=VALID_PHYSIOTHERAPY.index(result.get('had_physiotherapy', 'no')))
                    info_needed = st.selectbox('Further Info Needed', VALID_FURTHER_INFO, index=VALID_FURTHER_INFO.index(result.get('further_information_needed', 'no')))
                
                save_col, cancel_col = st.columns(2)
                with save_col:
                    if st.form_submit_button('💾 Save Modifications', use_container_width=True, type="primary"):
                        modified_result = result.copy()
                        modified_result.update({
                            'procedure_type': proc_type, 
                            'body_part': body_part, 
                            'arthroplasty_type': arthro_type, 
                            'had_injections': injections, 
                            'had_physiotherapy': physio, 
                            'further_information_needed': info_needed
                        })
                        
                        if save_analysis_changes(result, modified_result, analysis_id, uploaded_file.name):
                            st.toast("Correction saved!", icon="👍")
                        
                        save_analysis_history(modified_result, analysis_id, uploaded_file.name, "human_validated")
                        st.session_state[f"analyzed_{uploaded_file.name}"]['result'] = modified_result
                        st.session_state[f'modify_{analysis_id}'] = False
                        st.success("Classification updated.")
                        time.sleep(1)
                        st.rerun()
                
                with cancel_col:
                    if st.form_submit_button('Cancel', use_container_width=True):
                        st.session_state[f'modify_{analysis_id}'] = False
                        st.rerun()

        # Action Buttons
        st.markdown('<div class="card-actions">', unsafe_allow_html=True)
        b_col1, b_col2, b_col3, b_col4 = st.columns(4)
        
        with b_col1:
            st.button("Accept", key=f"accept_{analysis_id}", use_container_width=True)
        
        with b_col2:
            if st.button("Modify", key=f"modify_{analysis_id}", use_container_width=True):
                st.session_state[f'modify_{analysis_id}'] = not st.session_state.get(f'modify_{analysis_id}', False)
                st.rerun()
        
        with b_col3:
            st.button("Request Info", key=f"info_{analysis_id}", use_container_width=True)
        
        with b_col4:
            st.button("Export", key=f"export_{analysis_id}", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()