import os
import tempfile
import PyPDF2
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from pydantic import BaseModel, Field
import asyncio
import logging
import time
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from langchain.llms import Ollama
import json
import traceback
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
import re
import uuid
import pandas as pd
from datetime import datetime
from typing import Dict, Union, TypedDict, Optional
from functools import partial

logging.basicConfig(level=logging.INFO)

# Initialize the OCR model
model = ocr_predictor(pretrained=True)

# Initialize the Llama model
llm = Ollama(model="llama3.1:70b")

# Valid values for medical referral fields
VALID_PROCEDURE_TYPES = ['arthroplasty', 'soft tissue', 'unknown']
VALID_BODY_PARTS = ['hip', 'knee', 'unknown']
VALID_ARTHROPLASTY_TYPES = ['primary', 'revision', 'na', 'unknown']
VALID_FURTHER_INFO = ['yes', 'no']

class MedicalReferral(BaseModel):
    procedure_type: str = Field(description="The type of procedure (arthroplasty, soft tissue)")
    body_part: str = Field(description="The body part involved (hip, knee)")
    arthroplasty_type: str = Field(description="The type of arthroplasty (primary, revision)")
    further_information_needed: str = Field(description="Whether further information is needed (yes, no)")

class AnalysisResult(TypedDict):
    procedure_type: str
    body_part: str
    arthroplasty_type: str
    further_information_needed: str
    confidence: float

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
        
        text = ""
        for i, image in enumerate(images):
            # Perform OCR on each image
            page_text = pytesseract.image_to_string(image)
            print(f"\nPage {i+1} Content:")
            print("="*50)
            print(page_text)
            print("="*50)
            print(f"Characters extracted: {len(page_text)}")
            text += page_text + "\n"
        
        if not text.strip():
            logging.warning(f"No text extracted from {pdf_file.name} even after OCR.")
            return "Error: Unable to extract text from PDF. The document might be scanned without OCR or protected."
        else:
            logging.info(f"Successfully extracted {len(text)} characters using OCR")
        
        return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF {pdf_file.name}: {str(e)}")
        logging.error(traceback.format_exc())
        return f"Error: {str(e)}"
    finally:
        os.unlink(temp_file_path)
        end_time = time.time()
        print(f"Text extraction completed in {end_time - start_time:.2f} seconds")

# Rule-based analysis
def analyze_medical_referral(text) -> AnalysisResult:
    print("Starting medical referral analysis...")
    start_time = time.time()
    try:
        # Initialize with unknowns and yes for further info needed
        result = {
            "procedure_type": "unknown",
            "body_part": "unknown",
            "arthroplasty_type": "unknown",
            "further_information_needed": "yes",
            "confidence": 0.0
        }
        
        text = text.lower()
        
        # Determine body part
        if any(term in text for term in ['hip', 'acetabul', 'femoral head']):
            result["body_part"] = 'hip'
        elif any(term in text for term in ['knee', 'patella', 'tibia', 'femur']):
            result["body_part"] = 'knee'
        
        # Enhanced revision detection first
        previous_replacement_indicators = [
            'had a knee replacement', 'had a hip replacement',
            'previous replacement', 'previous arthroplasty',
            'post replacement', 'post arthroplasty',
            'after replacement', 'after arthroplasty',
            'bilateral.*replacements',  # Add this pattern
            'previously had.*replacements',  # Add this pattern
            'replacements.*in situ',  # Add this pattern
            'had.*bilateral'  # Add this pattern
        ]
        
        if any(re.search(indicator, text) for indicator in previous_replacement_indicators):
            result["procedure_type"] = 'arthroplasty'
            result["arthroplasty_type"] = 'revision'
        else:
            # Original arthroplasty detection
            arthroplasty_indicators = [
                'replacement', 'arthroplasty',
                'consideration for hip replacement', 'consideration for knee replacement',
                'moderate oa', 'severe oa', 'advanced oa',
                'not coping', 'mobility seriously limited',
                'failed conservative', 'failed physio', 'despite physiotherapy',
                'degenerative changes', 'loss of joint space',  # Add these indicators
                'osteophytes', 'walks with a frame',           # Add these indicators
                'debilitating'                                 # Add this indicator
            ]
            
            # Add soft tissue conditions including knee-specific ones
            soft_tissue_indicators = [
                'bursitis', 'impingement', 'tendinopathy',
                'tendinitis', 'tendonitis', 'labral',
                'trochanteric', 'abductor',
                'meniscal tear', 'meniscus tear',  # Add these specific indicators
                'acl', 'mcl', 'lcl', 'pcl',
                'ligament tear', 'cartilage tear'
            ]
            
            if any(indicator in text for indicator in soft_tissue_indicators):
                result["procedure_type"] = 'soft tissue'
            elif any(indicator in text for indicator in arthroplasty_indicators):
                result["procedure_type"] = 'arthroplasty'
                result["arthroplasty_type"] = 'primary'
        
        # Add confidence based on completeness
        if (result["body_part"] != "unknown" and 
            result["procedure_type"] != "unknown" and 
            (result["procedure_type"] != "arthroplasty" or 
             (result["procedure_type"] == "arthroplasty" and result["arthroplasty_type"] != "unknown"))):
            result["confidence"] = 0.8
        else:
            result["confidence"] = 0.3
        
        is_valid, error_msg = validate_analysis_result(result)
        if not is_valid:
            logging.error(f"Invalid analysis result: {error_msg}")
            return {
                "procedure_type": "unknown",
                "body_part": "unknown",
                "arthroplasty_type": "unknown",
                "further_information_needed": "yes",
                "confidence": 0.0
            }
            
        return result

    except Exception as e:
        logging.error(f"An error occurred in analyze_medical_referral: {str(e)}")
        return {
            "procedure_type": "unknown",
            "body_part": "unknown",
            "arthroplasty_type": "unknown",
            "further_information_needed": "yes",
            "confidence": 0.0
        }
    finally:
        end_time = time.time()
        print(f"Medical referral analysis completed in {end_time - start_time:.2f} seconds")

def analyze_medical_referral_with_llama(text):
    analysis_id = str(uuid.uuid4())[:8]
    print(f"\nStarting medical referral analysis with Llama... (ID: {analysis_id})")
    start_time = time.time()
    
    try:
        if not text.strip():
            return {
                "procedure_type": "unknown",
                "body_part": "unknown",
                "arthroplasty_type": "unknown",
                "further_information_needed": "yes",
                "confidence": 0.0
            }

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)
        print(f"\nAnalyzing {len(chunks)} chunks of text (ID: {analysis_id})")

        results = []
        for chunk_num, chunk in enumerate(chunks[:10], 1):
            print(f"\nChunk {chunk_num}/10 (ID: {analysis_id})")
            print("="*50)
            print("CHUNK CONTENT:")
            print(chunk)
            print("="*50)
            
            prompt = f"""Analyze this medical referral text with special attention to treatment history and symptoms:

TEXT TO ANALYZE:
{chunk}

CLASSIFICATION RULES:
1. Body Part (hip or knee):
- Look for specific mentions (e.g., "left knee", "right hip")
- Check for associated symptoms and movements (e.g., "stairs" for knee)

2. Procedure Type:
- If conservative treatment is ongoing/recent = "soft_tissue"
- If conservative treatment failed = "arthroplasty"
- If unclear treatment path = "unknown"

3. Arthroplasty Type:
- If previous/failed replacement = "revision"
- If first-time consideration = "primary"
- If not arthroplasty = "na"
- If unclear = "unknown"

4. Further Information:
- If treatment path and severity unclear = "yes"
- If clear progression of treatment = "no"

5. Confidence:
- High (0.8-1.0): Clear body part + clear treatment history
- Medium (0.5-0.7): Clear body part + partial treatment info
- Low (0.1-0.4): Unclear or conflicting information

RESPOND WITH THIS EXACT JSON FORMAT:
{{
    "procedure_type": "arthroplasty/soft_tissue/unknown",
    "body_part": "knee/hip/unknown",
    "arthroplasty_type": "primary/revision/na/unknown",
    "further_information_needed": "yes/no",
    "confidence": 0.0-1.0
}}"""

            print("\nSending to Llama for analysis...")
            response = llm.invoke(prompt)
            print(f"\nLlama response for chunk {chunk_num}:")
            print("-"*50)
            print(response)
            print("-"*50)
            
            try:
                json_match = re.search(r'\{.*?\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    json_str = re.sub(r'[\n\r\t]', '', json_str)
                    json_str = re.sub(r',\s*}', '}', json_str)
                    
                    result = json.loads(json_str)
                    
                    # Normalize values
                    result = {
                        "procedure_type": result["procedure_type"].lower().replace("_", " ").strip(),
                        "body_part": result["body_part"].lower().strip(),
                        "arthroplasty_type": result["arthroplasty_type"].lower().strip(),
                        "further_information_needed": result["further_information_needed"].lower().strip(),
                        "confidence": float(result.get("confidence", 0.0))
                    }
                    
                    # Validate values
                    if result["procedure_type"] not in VALID_PROCEDURE_TYPES:
                        result["procedure_type"] = "unknown"
                    if result["body_part"] not in VALID_BODY_PARTS:
                        result["body_part"] = "unknown"
                    if result["arthroplasty_type"] not in VALID_ARTHROPLASTY_TYPES:
                        result["arthroplasty_type"] = "unknown"
                    if result["further_information_needed"] not in VALID_FURTHER_INFO:
                        result["further_information_needed"] = "yes"
                        
                    results.append(result)
            except Exception as e:
                print(f"Error processing chunk {chunk_num}: {str(e)}")

        if not results:
            return {
                "procedure_type": "unknown",
                "body_part": "unknown",
                "arthroplasty_type": "unknown",
                "further_information_needed": "yes",
                "confidence": 0.0
            }

        # Combine results with enhanced logic
        combined_result = {
            "procedure_type": "unknown",
            "body_part": "unknown",
            "arthroplasty_type": "unknown",
            "further_information_needed": "yes",
            "confidence": 0.0
        }
        
        # Get most common non-unknown values
        body_parts = [r['body_part'] for r in results if r['body_part'] != "unknown"]
        if body_parts:
            combined_result["body_part"] = max(set(body_parts), key=body_parts.count)
            
        procedures = [r['procedure_type'] for r in results if r['procedure_type'] != "unknown"]
        if procedures:
            if "arthroplasty" in procedures:
                combined_result["procedure_type"] = "arthroplasty"
                
                # Look for arthroplasty type
                arthroplasty_types = [r['arthroplasty_type'] for r in results 
                                    if r['procedure_type'] == "arthroplasty" 
                                    and r['arthroplasty_type'] not in ["unknown", "na"]]
                if arthroplasty_types:
                    if "revision" in arthroplasty_types:
                        combined_result["arthroplasty_type"] = "revision"
                    else:
                        combined_result["arthroplasty_type"] = "primary"
            else:
                combined_result["procedure_type"] = max(set(procedures), key=procedures.count)
                combined_result["arthroplasty_type"] = "na"
        
        # Set confidence based on completeness
        if (combined_result["body_part"] != "unknown" and 
            combined_result["procedure_type"] != "unknown" and
            combined_result["arthroplasty_type"] != "unknown"):
            combined_result["confidence"] = max(r["confidence"] for r in results)
            combined_result["further_information_needed"] = "no"
        
        return combined_result

    except Exception as e:
        logging.error(f"Error in analyze_medical_referral_with_llama: {str(e)}")
        logging.error(traceback.format_exc())
        return {
            "procedure_type": "unknown",
            "body_part": "unknown",
            "arthroplasty_type": "unknown",
            "further_information_needed": "yes",
            "confidence": 0.0
        }
    finally:
        end_time = time.time()
        print(f"Analysis completed in {end_time - start_time:.2f} seconds (ID: {analysis_id})")


def save_analysis_changes(original_result, modified_result, analysis_id, filename):
    """Save all field values, marking which ones were changed"""
    
    # Validate both results
    is_valid_orig, error_msg = validate_analysis_result(original_result)
    if not is_valid_orig:
        logging.error(f"Invalid original result: {error_msg}")
        return False
        
    is_valid_mod, error_msg = validate_analysis_result(modified_result)
    if not is_valid_mod:
        logging.error(f"Invalid modified result: {error_msg}")
        return False
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Check if any changes were made
    changes_made = False
    for key in original_result:
        if original_result[key] != modified_result[key]:
            changes_made = True
            break
    
    if changes_made:
        data = []
        # Save all fields
        for key in original_result:
            data.append({
                'timestamp': timestamp,
                'analysis_id': analysis_id,
                'filename': filename,
                'field': key,
                'original_value': original_result[key],
                'modified_value': modified_result[key],
                'was_changed': original_result[key] != modified_result[key]
            })
        
        df = pd.DataFrame(data)
        csv_path = 'datasets/analysis_changes.csv'
        
        # Append to existing CSV or create new one
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            # Add 'was_changed' column to track which fields were actually modified
            df.to_csv(csv_path, index=False)
        
        return True
    return False

def calculate_accuracy_metrics():
    """Calculate accuracy metrics from saved changes"""
    if not os.path.exists('datasets/analysis_history.csv'):
        return {
            'total_analyzed': 0,
            'total_corrected': 0,
            'accuracy_by_field': {},
            'overall_accuracy': 0.0
        }
    
    try:
        # Update paths to include 'datasets/' directory
        history_df = pd.read_csv('datasets/analysis_history.csv')
        total_analyzed = len(history_df['analysis_id'].unique())
        
        # Read corrections (if any exist)
        changes_path = 'datasets/analysis_changes.csv'
        if os.path.exists(changes_path):
            changes_df = pd.read_csv(changes_path)
            total_corrected = len(changes_df['analysis_id'].unique())
        else:
            total_corrected = 0
            changes_df = pd.DataFrame()
        
        # Calculate accuracy by field
        field_counts = changes_df['field'].value_counts() if not changes_df.empty else pd.Series()
        all_fields = ['procedure_type', 'body_part', 'arthroplasty_type', 'further_information_needed']
        
        accuracy_by_field = {}
        for field in all_fields:
            errors = field_counts.get(field, 0)
            accuracy = (total_analyzed - errors) / total_analyzed if total_analyzed > 0 else 0
            accuracy_by_field[field] = round(accuracy * 100, 2)
        
        # Calculate overall accuracy
        overall_accuracy = (total_analyzed - total_corrected) / total_analyzed if total_analyzed > 0 else 0
        overall_accuracy = round(overall_accuracy * 100, 2)
        
        return {
            'total_analyzed': total_analyzed,
            'total_corrected': total_corrected,
            'accuracy_by_field': accuracy_by_field,
            'overall_accuracy': overall_accuracy
        }
    except Exception as e:
        logging.error(f"Error calculating accuracy metrics: {str(e)}")
        return None

def save_analysis_history(result, analysis_id, filename, model_type):
    """Save original analysis results"""
    try:
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            'analysis_id': [analysis_id],
            'filename': [filename],
            'model_type': [model_type],  # 'llama' or 'rule_based'
            'procedure_type': [result['procedure_type']],
            'body_part': [result['body_part']],
            'arthroplasty_type': [result['arthroplasty_type']],
            'further_information_needed': [result['further_information_needed']]
        })
        
        # Validate DataFrame
        is_valid, error_msg = validate_csv_data(df, 'history')
        if not is_valid:
            logging.error(f"Invalid CSV data: {error_msg}")
            return False
        
        # Continue with existing save logic...
        
    except Exception as e:
        logging.error(f"Error saving analysis history: {str(e)}")
        return False

def determine_pattern(procedure_type: str, body_part: str, arthroplasty_type: str) -> str:
    """Determine if the analysis pattern is clear or unclear"""
    if (procedure_type.lower() == "unknown" or 
        body_part.lower() not in ["hip", "knee"] or 
        (procedure_type.lower() == "arthroplasty" and arthroplasty_type.lower() == "unknown")):
        return "unclear pathology"
    return "clear"

def validate_analysis_result(result: dict) -> tuple[bool, Optional[str]]:
    """
    Validate analysis result structure and values.
    Returns (is_valid, error_message)
    """
    required_fields = {
        'procedure_type': VALID_PROCEDURE_TYPES,
        'body_part': VALID_BODY_PARTS,
        'arthroplasty_type': VALID_ARTHROPLASTY_TYPES,
        'further_information_needed': VALID_FURTHER_INFO,
        'confidence': float
    }
    
    try:
        # Check all required fields exist
        for field in required_fields:
            if field not in result:
                return False, f"Missing required field: {field}"
        
        # Validate field values
        for field, valid_values in required_fields.items():
            if field == 'confidence':
                if not isinstance(result[field], (int, float)):
                    return False, f"Invalid confidence value: must be numeric"
                if not 0 <= result[field] <= 1:
                    return False, f"Invalid confidence value: must be between 0 and 1"
            else:
                if result[field].lower() not in valid_values:
                    return False, f"Invalid value for {field}: {result[field]}"
        
        # Additional validation rules
        if result['procedure_type'] != 'arthroplasty' and result['arthroplasty_type'] != 'na':
            return False, "Arthroplasty type must be 'na' for non-arthroplasty procedures"
            
        return True, None
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def validate_csv_data(df: pd.DataFrame, csv_type: str) -> tuple[bool, Optional[str]]:
    """
    Validate DataFrame before saving to CSV
    csv_type can be 'history' or 'changes'
    """
    try:
        if csv_type == 'history':
            required_columns = {
                'timestamp': str,
                'analysis_id': str,
                'filename': str,
                'model_type': str,
                'procedure_type': str,
                'body_part': str,
                'arthroplasty_type': str,
                'further_information_needed': str
            }
        else:  # changes
            required_columns = {
                'timestamp': str,
                'analysis_id': str,
                'filename': str,
                'field': str,
                'original_value': str,
                'modified_value': str,
                'was_changed': bool
            }
        
        # Check all required columns exist
        missing_cols = set(required_columns.keys()) - set(df.columns)
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
        
        # Check for null values
        null_counts = df.isnull().sum()
        if null_counts.any():
            return False, f"Found null values in columns: {null_counts[null_counts > 0].index.tolist()}"
        
        # Validate data types
        for col, expected_type in required_columns.items():
            if not all(isinstance(val, expected_type) for val in df[col].dropna()):
                return False, f"Invalid data type in column {col}"
        
        return True, None
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def ensure_data_directories():
    """Create necessary directories and files if they don't exist"""
    # Create datasets directory if it doesn't exist
    if not os.path.exists('datasets'):
        os.makedirs('datasets')
    
    # Create initial CSV files with headers if they don't exist
    history_headers = [
        'timestamp', 'analysis_id', 'filename', 'model_type',
        'procedure_type', 'body_part', 'arthroplasty_type',
        'further_information_needed'
    ]
    
    changes_headers = [
        'timestamp', 'analysis_id', 'filename', 'field',
        'original_value', 'modified_value', 'was_changed'
    ]
    
    actions_headers = [
        'timestamp', 'filename', 'info_requested', 'referral_accepted'
    ]
    
    # Create analysis_history.csv if it doesn't exist
    history_path = 'datasets/analysis_history.csv'
    if not os.path.exists(history_path):
        pd.DataFrame(columns=history_headers).to_csv(history_path, index=False)
    
    # Create analysis_changes.csv if it doesn't exist
    changes_path = 'datasets/analysis_changes.csv'
    if not os.path.exists(changes_path):
        pd.DataFrame(columns=changes_headers).to_csv(changes_path, index=False)
        
    # Create referral_actions.csv if it doesn't exist
    actions_path = 'datasets/referral_actions.csv'
    if not os.path.exists(actions_path):
        pd.DataFrame(columns=actions_headers).to_csv(actions_path, index=False)

def save_referral_action(filename: str, action_type: str):
    """Save referral action to CSV
    action_type can be 'info_requested' or 'referral_accepted'
    """
    try:
        actions_path = 'datasets/referral_actions.csv'
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Read existing actions if file exists
        if os.path.exists(actions_path):
            actions_df = pd.read_csv(actions_path)
        else:
            actions_df = pd.DataFrame(columns=['timestamp', 'filename', 'info_requested', 'referral_accepted'])
        
        # Check if file already exists in the DataFrame
        existing_row = actions_df[actions_df['filename'] == filename]
        
        if existing_row.empty:
            # Create new row
            new_row = {
                'timestamp': timestamp,
                'filename': filename,
                'info_requested': action_type == 'info_requested',
                'referral_accepted': action_type == 'referral_accepted'
            }
            actions_df = pd.concat([actions_df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            # Update existing row
            idx = existing_row.index[0]
            actions_df.at[idx, action_type] = True
            actions_df.at[idx, 'timestamp'] = timestamp
        
        # Save to CSV
        actions_df.to_csv(actions_path, index=False)
        return True
        
    except Exception as e:
        logging.error(f"Error saving referral action: {str(e)}")
        return False

@st.cache_data
def cached_analysis(text, analysis_id):
    return analyze_medical_referral_with_llama(text)

# Streamlit UI
def main():
    # Ensure data directories and files exist
    ensure_data_directories()
    
    # Page configuration
    st.set_page_config(
        page_title="Orthopedic Referral Triage Assistant",
        page_icon="🏥",
        layout="wide"
    )
    
    st.title("🏥 Orthopedic Referral Triage Assistant")
    
    # Purpose statement
    st.markdown("""
    ### Purpose
    This AI-powered tool helps classify and prioritize orthopedic referrals for hip and knee conditions.
    
    **Capabilities:**
    - 📄 Extracts text from PDF referral letters
    - 🔍 Identifies body part and procedure type
    - 📊 Tracks analysis accuracy
    """)

    # Clear cached data
    st.cache_data.clear()

    # Create tabs
    tab1, tab2 = st.tabs(["📝 New Referrals", "📊 Analysis Details"])

    with tab1:
        uploaded_files = st.file_uploader(
            "Upload Referral Letters (PDF)", 
            accept_multiple_files=True, 
            type="pdf",
            help="Upload one or more PDF referral letters for analysis"
        )

        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_key = f"analyzed_{uploaded_file.name}"
                
                # Only run analysis if we haven't already analyzed this file
                if file_key not in st.session_state:
                    with st.spinner('Processing document...'):
                        try:
                            # Extract text and run analysis
                            text = asyncio.run(extract_text_from_pdf(uploaded_file))
                            if text.startswith("Error:"):
                                st.error(text)
                                continue

                            analysis_id = str(uuid.uuid4())[:8]
                            result = cached_analysis(text, analysis_id)
                            
                            # Store results in session state
                            st.session_state[file_key] = {
                                'text': text,
                                'analysis_id': analysis_id,
                                'result': result,
                                'priority': "Routine"  # Default priority
                            }
                        except Exception as e:
                            st.error(f"Error processing file: {str(e)}")
                            logging.error(traceback.format_exc())
                            continue

                try:
                    # Get stored results
                    analysis_data = st.session_state[file_key]
                    text = analysis_data['text']
                    analysis_id = analysis_data['analysis_id']
                    result = analysis_data['result']

                    with st.container():
                        st.markdown(f"### 📄 Analyzing: {uploaded_file.name}")
                        
                        # Display results in columns
                        col1, col2, col3 = st.columns([1.5, 1.5, 2])
                        
                        with col1:
                            st.markdown("#### 📋 Quick Summary")
                            summary_data = pd.DataFrame({
                                "Field": [
                                    "Body Part",
                                    "Procedure Type",
                                    "Arthroplasty Type",
                                    "Needs More Info"
                                ],
                                "Value": [
                                    result["body_part"].title(),
                                    result["procedure_type"].title(),
                                    "Primary" if result["arthroplasty_type"] == "primary"
                                    else "Revision" if result["arthroplasty_type"] == "revision"
                                    else "N/A" if result["arthroplasty_type"] == "na"
                                    else "Unknown" if result["procedure_type"] == "arthroplasty"
                                    else "N/A",
                                    result["further_information_needed"].title()
                                ]
                            })
                            st.table(summary_data)
                            
                            # Confidence indicator
                            confidence = float(result.get('confidence', 0.0))
                            st.progress(confidence, text=f"Confidence: {confidence:.1%}")

                        with col2:
                            # st.markdown("#### 🎯 Priority Setting")
                            
                            # Comment out priority setting
                            # priority = st.select_slider(
                            #     "Set Priority Level",
                            #     options=["Urgent", "Soon", "Routine"],
                            #     value=st.session_state[file_key].get('priority', "Routine"),
                            #     key=f"priority_{analysis_id}",
                            #     help="Drag to set the priority level for this referral"
                            # )
                            # st.session_state[file_key]['priority'] = priority
                            
                            st.markdown("#### ⚡ Quick Actions")
                            
                            action_col1, action_col2 = st.columns(2)
                            with action_col1:
                                if st.button(
                                    "📧 Request Info",
                                    key=f"request_btn_{analysis_id}",
                                    help="Generate an information request email"
                                ):
                                    if save_referral_action(uploaded_file.name, 'info_requested'):
                                        st.success("Information request recorded")
                                    else:
                                        st.error("Error recording information request")
                            
                            with action_col2:
                                if st.button(
                                    "✅ Accept Referral",
                                    key=f"accept_btn_{analysis_id}",
                                    help="Process this referral"
                                ):
                                    if save_referral_action(uploaded_file.name, 'referral_accepted'):
                                        st.success("Referral acceptance recorded")
                                    else:
                                        st.error("Error recording referral acceptance")
                        
                        with col3:
                            st.markdown("#### ✏️ Modify Classification")
                            
                            # Create modification form
                            with st.form(key=f"modify_form_{analysis_id}"):
                                modified_result = {}
                                
                                # Use the stored result for initial values
                                stored_result = st.session_state[file_key]['result']
                                
                                # Procedure Type
                                modified_result['procedure_type'] = st.selectbox(
                                    'Procedure Type',
                                    options=VALID_PROCEDURE_TYPES,
                                    index=VALID_PROCEDURE_TYPES.index(
                                        st.session_state[file_key]['result']['procedure_type'].lower().replace("_", " ")
                                    ),
                                    help="Select the main procedure type"
                                )
                                
                                # Body Part
                                modified_result['body_part'] = st.selectbox(
                                    'Body Part',
                                    options=VALID_BODY_PARTS,
                                    index=VALID_BODY_PARTS.index(
                                        st.session_state[file_key]['result']['body_part'].lower()
                                    ),
                                    help="Select the affected body part"
                                )
                                
                                # Arthroplasty Type
                                if modified_result['procedure_type'] == 'arthroplasty':
                                    valid_types = [t for t in VALID_ARTHROPLASTY_TYPES if t != "na"]
                                    current_type = st.session_state[file_key]['result']['arthroplasty_type']
                                    if current_type == "na":
                                        current_type = "unknown"
                                    
                                    modified_result['arthroplasty_type'] = st.selectbox(
                                        'Arthroplasty Type',
                                        options=valid_types,
                                        index=valid_types.index(current_type),
                                        help="Select the type of arthroplasty"
                                    )
                                else:
                                    modified_result['arthroplasty_type'] = "na"
                                
                                # Further Information
                                modified_result['further_information_needed'] = st.selectbox(
                                    'Further Information Needed',
                                    options=VALID_FURTHER_INFO,
                                    index=VALID_FURTHER_INFO.index(
                                        st.session_state[file_key]['result']['further_information_needed']
                                    ),
                                    help="Indicate if more information is needed"
                                )
                                
                                # Add confidence from original result
                                modified_result['confidence'] = st.session_state[file_key]['result']['confidence']
                                
                                # Save button
                                if st.form_submit_button('Save Changes'):
                                    changes_saved = save_analysis_changes(
                                        stored_result,
                                        modified_result,
                                        analysis_id,
                                        uploaded_file.name
                                    )
                                    if changes_saved:
                                        # Update stored result
                                        st.session_state[file_key]['result'] = modified_result
                                        st.success('✅ Changes saved successfully!')
                                    else:
                                        st.info('ℹ️ No changes detected.')

                        # Show extracted text in expandable section
                        with st.expander("📝 View Extracted Text", expanded=False):
                            st.text_area("", text, height=200)

                except Exception as e:
                    st.error(f"Error displaying results: {str(e)}")
                    logging.error(traceback.format_exc())

    # Statistics Tab
    with tab2:
        display_analysis_statistics()  # This function should be implemented separately

def display_analysis_statistics():
    """Display analysis statistics and metrics in the Statistics tab"""
    
    # Get accuracy metrics
    metrics = calculate_accuracy_metrics()
    
    if not metrics:
        st.warning("No analysis data available yet.")
        return
        
    # Create three columns for statistics
    col1, col2, col3 = st.columns(3)
    
    # Overall Statistics
    with col1:
        st.markdown("### 📊 Overall Statistics")
        st.metric(
            label="Total Referrals Analyzed",
            value=metrics['total_analyzed']
        )
        st.metric(
            label="Overall Accuracy",
            value=f"{metrics['overall_accuracy']}%",
            delta=f"{metrics['overall_accuracy'] - 50:.1f}%" if metrics['overall_accuracy'] > 0 else None
        )
        st.metric(
            label="Referrals Needing Correction",
            value=metrics['total_corrected'],
            delta=f"-{metrics['total_corrected']}" if metrics['total_corrected'] > 0 else None,
            delta_color="inverse"
        )

    # Field-specific Accuracy
    with col2:
        st.markdown("### 🎯 Accuracy by Field")
        field_accuracy = metrics['accuracy_by_field']
        
        # Create accuracy indicators for each field
        for field, accuracy in field_accuracy.items():
            # Format field name for display
            display_name = field.replace('_', ' ').title()
            
            # Color code based on accuracy
            color = (
                "🟢" if accuracy >= 90 else
                "🟡" if accuracy >= 75 else
                "🔴"
            )
            
            st.markdown(f"{color} **{display_name}:** {accuracy}%")

    # Historical Trend
    with col3:
        st.markdown("### 📈 Analysis History")
        try:
            # Read the history file
            history_df = pd.read_csv('datasets/analysis_history.csv')
            
            if not history_df.empty:
                # Convert timestamp to datetime
                history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
                
                # Group by date and count analyses
                daily_counts = (
                    history_df.groupby(history_df['timestamp'].dt.date)
                    .size()
                    .reset_index(name='count')
                )
                
                # Create line chart
                st.line_chart(
                    daily_counts.set_index('timestamp')['count'],
                    use_container_width=True
                )
                
                # Show recent activity
                st.markdown("#### Recent Activity")
                recent = history_df.tail(5)[['timestamp', 'filename', 'model_type']]
                st.dataframe(recent, use_container_width=True)
            else:
                st.info("No historical data available yet.")
                
        except Exception as e:
            st.error(f"Error loading historical data: {str(e)}")
    
    # Additional Analysis Details
    st.markdown("### 🔍 Detailed Analysis")
    
    try:
        # Read both history and changes files
        history_df = pd.read_csv('analysis/analysis_history.csv')
        changes_df = pd.read_csv('analysis/analysis_changes.csv') if os.path.exists('analysis/analysis_changes.csv') else pd.DataFrame()
        
        tab1, tab2 = st.tabs(["📝 Analysis Distribution", "⚠️ Common Corrections"])
        
        with tab1:
            if not history_df.empty:
                # Create distribution charts
                col1, col2 = st.columns(2)
                
                with col1:
                    # Procedure Type Distribution
                    proc_dist = history_df['procedure_type'].value_counts()
                    st.markdown("#### Procedure Types")
                    st.bar_chart(proc_dist)
                
                with col2:
                    # Body Part Distribution
                    body_dist = history_df['body_part'].value_counts()
                    st.markdown("#### Body Parts")
                    st.bar_chart(body_dist)
            else:
                st.info("No analysis data available yet.")
        
        with tab2:
            if not changes_df.empty:
                # Analyze common corrections
                corrections = changes_df[changes_df['was_changed']]
                common_corrections = (
                    corrections.groupby(['field', 'original_value', 'modified_value'])
                    .size()
                    .reset_index(name='count')
                    .sort_values('count', ascending=False)
                    .head(10)
                )
                
                st.markdown("#### Most Common Corrections")
                st.dataframe(
                    common_corrections,
                    use_container_width=True,
                    column_config={
                        "field": "Field",
                        "original_value": "Original Value",
                        "modified_value": "Corrected To",
                        "count": "Frequency"
                    }
                )
            else:
                st.info("No corrections data available yet.")
    
    except Exception as e:
        st.error(f"Error analyzing detailed statistics: {str(e)}")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()
