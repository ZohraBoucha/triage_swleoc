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

logging.basicConfig(level=logging.INFO)

# Initialize the OCR model
model = ocr_predictor(pretrained=True)

# Initialize the Llama model
llm = Ollama(model="llama3.1:70b")

class MedicalReferral(BaseModel):
    procedure_type: str = Field(description="The type of procedure (arthroplasty, soft tissue)")
    body_part: str = Field(description="The body part involved (hip, knee)")
    arthroplasty: str = Field(description="The type of arthroplasty (primary, revision)")
    further_information_needed: str = Field(description="Whether further information is needed (yes, no)")

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
            print(f"Page {i+1}: Extracted {len(page_text)} characters")
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

def analyze_medical_referral(text):
    print("Starting medical referral analysis...")
    start_time = time.time()
    try:
        # Initialize with unknowns and yes for further info needed
        result = MedicalReferral(
            procedure_type='unknown',
            body_part='unknown',
            arthroplasty='unknown',
            further_information_needed='yes'  # Start with yes
        )
        
        text = text.lower()
        
        # Determine body part
        if any(term in text for term in ['hip', 'acetabul', 'femoral head']):
            result.body_part = 'hip'
        elif any(term in text for term in ['knee', 'patella', 'tibia', 'femur']):
            result.body_part = 'knee'
        
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
            result.procedure_type = 'arthroplasty'
            result.arthroplasty = 'revision'
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
                result.procedure_type = 'soft tissue'
            elif any(indicator in text for indicator in arthroplasty_indicators):
                result.procedure_type = 'arthroplasty'
                result.arthroplasty = 'primary'
        
        # Only set to 'no' if ALL required fields are known
        if (result.body_part != 'unknown' and 
            result.procedure_type != 'unknown' and 
            (result.procedure_type != 'arthroplasty' or 
             (result.procedure_type == 'arthroplasty' and result.arthroplasty != 'unknown'))):
            result.further_information_needed = 'no'
        else:
            result.further_information_needed = 'yes'
        
        return result

    except Exception as e:
        logging.error(f"An error occurred in analyze_medical_referral: {str(e)}")
        return f"Error: {str(e)}"
    finally:
        end_time = time.time()
        print(f"Medical referral analysis completed in {end_time - start_time:.2f} seconds")

def analyze_medical_referral_with_llama(text):
    analysis_id = str(uuid.uuid4())[:8]
    print(f"Starting medical referral analysis with Llama... (ID: {analysis_id})")
    start_time = time.time()
    try:
        if not text.strip():
            return f"Error: No text provided for analysis (ID: {analysis_id})"

        print(f"Text to analyze (first 200 chars): {text[:200]} (ID: {analysis_id})")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)

        print(f"Analyzing {len(chunks)} chunks of text (ID: {analysis_id})")

        results = []
        for chunk_num, chunk in enumerate(chunks[:10], 1):
            print(f"Analyzing chunk {chunk_num}/10 (ID: {analysis_id})")
            prompt = f"""Analyze this medical text:
            {chunk}

            Provide analysis in JSON:
            {{
                "procedure_type": "arthroplasty or soft tissue or unknown",
                "body_part": "hip or knee or unknown",
                "arthroplasty": "primary or revision or N/A or unknown",
                "further_information_needed": "yes or no or unknown"
            }}
            """

            response = llm.invoke(prompt)
            print(f"Llama response for chunk {chunk_num} (ID: {analysis_id}): {response}")
            
            try:
                json_match = re.search(r'\{[^}]+\}', response)
                if json_match:
                    result = json.loads(json_match.group())
                    if all(key in result for key in ["procedure_type", "body_part", "arthroplasty", "further_information_needed"]):
                        results.append(result)
                        print(f"Valid result found in chunk {chunk_num} (ID: {analysis_id})")
                    else:
                        print(f"Incomplete JSON in chunk {chunk_num} (ID: {analysis_id})")
                else:
                    print(f"No valid JSON found in chunk {chunk_num} (ID: {analysis_id})")
            except json.JSONDecodeError:
                print(f"Failed to parse JSON for chunk {chunk_num} (ID: {analysis_id})")

        if not results:
            return f"Error: No valid results from Llama analysis (ID: {analysis_id})"

        # Combine results from all analyzed chunks - prioritize known values over unknowns
        combined_result = {
            "procedure_type": max(
                (r['procedure_type'] for r in results),
                key=lambda x: 0 if x == "unknown" else results.count(x)
            ),
            "body_part": max(
                (r['body_part'] for r in results),
                key=lambda x: 0 if x == "unknown" else results.count(x)
            ),
            "arthroplasty": max(
                (r['arthroplasty'] for r in results),
                key=lambda x: 0 if x == "unknown" else results.count(x)
            ),
            "further_information_needed": "yes" if any(r['further_information_needed'] == "yes" for r in results) else "no"
        }

        print(f"Combined result (ID: {analysis_id}): {combined_result}")
        
        # Return the dictionary instead of formatting it
        return combined_result

    except Exception as e:
        logging.error(f"An error occurred in analyze_medical_referral_with_llama (ID: {analysis_id}): {str(e)}")
        logging.error(traceback.format_exc())
        return f"Error: {str(e)} (ID: {analysis_id})"
    finally:
        end_time = time.time()
        print(f"Medical referral analysis with Llama completed in {end_time - start_time:.2f} seconds (ID: {analysis_id})")

def save_analysis_changes(original_result, modified_result, analysis_id, filename):
    """Save all field values, marking which ones were changed"""
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
        csv_path = 'analysis_changes.csv'
        
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
    if not os.path.exists('analysis_history.csv'):
        return {
            'total_analyzed': 0,
            'total_corrected': 0,
            'accuracy_by_field': {},
            'overall_accuracy': 0.0
        }
    
    try:
        # Read all analyses
        history_df = pd.read_csv('analysis_history.csv')
        total_analyzed = len(history_df['analysis_id'].unique())
        
        # Read corrections (if any exist)
        if os.path.exists('analysis_changes.csv'):
            changes_df = pd.read_csv('analysis_changes.csv')
            total_corrected = len(changes_df['analysis_id'].unique())
        else:
            total_corrected = 0
            changes_df = pd.DataFrame()
        
        # Calculate accuracy by field
        field_counts = changes_df['field'].value_counts() if not changes_df.empty else pd.Series()
        all_fields = ['procedure_type', 'body_part', 'arthroplasty', 'further_information_needed']
        
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
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    data = {
        'timestamp': [timestamp],
        'analysis_id': [analysis_id],
        'filename': [filename],
        'model_type': [model_type],  # 'llama' or 'rule_based'
        'procedure_type': [result['procedure_type']],
        'body_part': [result['body_part']],
        'arthroplasty': [result['arthroplasty']],
        'further_information_needed': [result['further_information_needed']]
    }
    
    df = pd.DataFrame(data)
    csv_path = 'analysis_history.csv'
    
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)

# Streamlit UI
def main():
    st.title("Medical Referral Analyzer")
    
    # Display accuracy metrics
    st.sidebar.title("Analysis Accuracy")
    metrics = calculate_accuracy_metrics()
    
    if metrics:
        st.sidebar.metric("Overall Accuracy", f"{metrics['overall_accuracy']}%")
        st.sidebar.markdown("### Accuracy by Field")
        for field, accuracy in metrics['accuracy_by_field'].items():
            st.sidebar.metric(field.replace('_', ' ').title(), f"{accuracy}%")
        
        st.sidebar.markdown("### Summary")
        st.sidebar.text(f"Total analyzed: {metrics['total_analyzed']}")
        st.sidebar.text(f"Total corrected: {metrics['total_corrected']}")

    # Clear any cached data
    st.cache_data.clear()

    use_llama = st.checkbox("Use Llama 3.1 for analysis")

    uploaded_files = st.file_uploader("Choose PDF files", accept_multiple_files=True, type="pdf")

    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.write(f"Processing: {uploaded_file.name}")
            
            with st.spinner('Analyzing the document...'):
                try:
                    text = asyncio.run(extract_text_from_pdf(uploaded_file))
                    
                    if text.startswith("Error:"):
                        st.error(text)
                        continue

                    # Generate analysis ID at the start of processing
                    analysis_id = str(uuid.uuid4())[:8]
                    
                    # Create two columns for layout
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.subheader("Document Analysis")
                        
                        # First section: Key Terms Found
                        with st.expander("Key Terms Found", expanded=True):
                            text_lower = text.lower()
                            
                            # Create sections for different categories with their terms and titles
                            categories = [
                                (body_terms := [
                                    # Hip specific terms
                                    'hip', 'acetabul', 'femoral head', 'greater trochanter', 'femoroacetabular',
                                    # Knee specific terms
                                    'knee', 'patella', 'tibia', 'femur', 'meniscus'
                                ], 
                                "Body Part Terms"),
                                (procedure_terms := [
                                    # Hip arthroplasty terms
                                    'hip replacement', 'hip arthroplasty', 'total hip', r'\btha\b',
                                    # Knee arthroplasty terms
                                    'knee replacement', 'knee arthroplasty', 'total knee', r'\btka\b',
                                    # General arthroplasty terms
                                    'arthroplasty', 'replacement',
                                    # Hip soft tissue terms
                                    'labral tear', 'hip impingement', 'trochanteric bursitis',
                                    # Knee soft tissue terms
                                    'meniscal tear', r'\bacl\b', r'\bmcl\b', r'\bpcl\b', r'\blcl\b'
                                ], 
                                "Procedure Terms"),
                                (status_terms := [
                                    'primary', 'revision', 'previous', 'failed', 'redo',
                                    'first time', 'initial', 'prior surgery'
                                ], 
                                "Status Terms")
                            ]
                            
                            # Display found terms with their context and categorization
                            for terms, title in categories:
                                matches = []
                                for term in terms:
                                    if term in text_lower:
                                        start = max(0, text_lower.find(term) - 50)
                                        end = min(len(text), text_lower.find(term) + len(term) + 50)
                                        context = text[start:end].strip()
                                        if not context.endswith('.'):
                                            context += '...'
                                        if not context.startswith('.'):
                                            context = '...' + context
                                        
                                        # Add categorization for procedure terms
                                        if title == "Procedure Terms":
                                            if any(x in term for x in ['replacement', 'arthroplasty', 'tha', 'tka']):
                                                term = f"{term} (Arthroplasty)"
                                            else:
                                                term = f"{term} (Soft Tissue)"
                                        elif title == "Body Part Terms":
                                            if any(x in term for x in ['hip', 'acetabul', 'femoral', 'trochanter']):
                                                term = f"{term} (Hip)"
                                            else:
                                                term = f"{term} (Knee)"
                                                
                                        matches.append((term.capitalize(), context))
                                
                                if matches:
                                    st.markdown(f"**{title}**")
                                    for term, match in matches:
                                        st.markdown(f"- **{term}**: '{match}'")
                                    st.markdown("---")

                        # Second section: Decision Logic
                        with st.expander("Decision Logic", expanded=True):
                            st.markdown("**Analysis Logic:**")
                            
                            # Enhanced body part detection
                            primary_body_part = None
                            clinical_evidence = None
                            procedure_type = "Unknown"
                            
                            # Count meaningful mentions of each body part
                            hip_indicators = {
                                'hip pain': 0,
                                'hip replacement': 0,
                                'hip arthroplasty': 0,
                                'hip oa': 0
                            }
                            
                            knee_indicators = {
                                'knee pain': 0,
                                'knee replacement': 0,
                                'knee arthroplasty': 0,
                                'knee oa': 0
                            }
                            
                            # Count meaningful mentions
                            for indicator in hip_indicators:
                                hip_indicators[indicator] = text_lower.count(indicator)
                            
                            for indicator in knee_indicators:
                                knee_indicators[indicator] = text_lower.count(indicator)
                            
                            # Determine primary body part based on meaningful mentions
                            hip_score = sum(hip_indicators.values())
                            knee_score = sum(knee_indicators.values())
                            
                            # Add weight for specific procedure mentions
                            if 'knee replacement' in text_lower:
                                knee_score += 2
                            if 'hip replacement' in text_lower:
                                hip_score += 2
                            
                            primary_body_part = "Knee" if knee_score > hip_score else "Hip"
                            
                            st.markdown(f"- **Body Part:** {primary_body_part}")
                            
                            # Show evidence for body part determination
                            if primary_body_part == "Knee":
                                for indicator, count in knee_indicators.items():
                                    if count > 0:
                                        st.markdown(f"  - Found '{indicator}' {count} times")
                            else:
                                for indicator, count in hip_indicators.items():
                                    if count > 0:
                                        st.markdown(f"  - Found '{indicator}' {count} times")
                            
                            # Determine procedure type with enhanced detection
                            arthroplasty_indicators = [
                                'replacement', 'arthroplasty',
                                'consideration for hip replacement',
                                'consideration for knee replacement',
                                'moderate oa', 'severe oa', 'advanced oa',
                                'not coping', 'mobility seriously limited',
                                'failed conservative', 'failed physio',
                                'despite physiotherapy', 'in agony',
                                'degenerative changes',  # Add this indicator
                                'degenerative cha'      # Add shortened version for partial matches
                            ]
                            
                            soft_tissue_only_indicators = [
                                'acute tear', 'recent injury', 'fresh injury',
                                'new tear', 'sports injury'
                            ]
                            
                            # Check for degenerative changes specifically
                            has_degenerative_changes = any(term in text_lower for term in ['degenerative changes', 'degenerative cha'])
                            has_acute_injury = any(term in text_lower for term in soft_tissue_only_indicators)
                            
                            if has_degenerative_changes and not has_acute_injury:
                                procedure_type = "Arthroplasty"
                                pattern = "degenerative joint disease"
                                intervention = "arthroplasty"
                            elif any(indicator in text_lower for indicator in arthroplasty_indicators):
                                procedure_type = "Arthroplasty"
                                pattern = "degenerative joint disease"
                                intervention = "arthroplasty"
                            else:
                                procedure_type = "Unknown"
                                pattern = "unclear pathology"
                                intervention = "further investigation needed"
                            
                            st.markdown(f"- **Procedure Type:** {procedure_type}")
                            if clinical_evidence:
                                st.markdown(f"  - Evidence: '{clinical_evidence}'")
                            
                            # Show evidence that led to the decision
                            if procedure_type == "Arthroplasty":
                                st.markdown("Evidence for arthroplasty:")
                                for indicator in arthroplasty_indicators:
                                    if indicator in text_lower:
                                        start = max(0, text_lower.find(indicator) - 50)
                                        end = min(len(text), text_lower.find(indicator) + len(indicator) + 50)
                                        context = text[start:end].strip()
                                        st.markdown(f"  - Found '{indicator}' in context: '{context}'")
                            
                            st.markdown(f"- Most appropriate intervention would be {intervention}")
                            
                            # Analyze urgency
                            urgency_indicators = [
                                'in agony', 'severe pain', 'seriously limited',
                                'not coping', 'failed conservative', 'despite treatment',
                                'mobility seriously limited', 'unable to cope'
                            ]
                            
                            is_urgent = any(indicator in text_lower for indicator in urgency_indicators)
                            st.markdown(f"- Level of urgency: {'urgent' if is_urgent else 'routine'}")
                            if is_urgent:
                                for indicator in urgency_indicators:
                                    if indicator in text_lower:
                                        st.markdown(f"  - Found '{indicator}'")

                    with col2:
                        print("Starting Analysis Results section...")
                        st.subheader("Analysis Results")
                        if use_llama:
                            print("Using Llama path...")
                            llama_result = analyze_medical_referral_with_llama(text)
                            if isinstance(llama_result, dict):  # Check if we got a valid dictionary result
                                original_result = llama_result  # Use the result directly
                                
                                # Create DataFrame and display
                                df = pd.DataFrame({
                                    'Field': ['Procedure Type', 'Body Part', 'Arthroplasty', 'Further Information Needed'],
                                    'Value': [
                                        original_result['procedure_type'].title(),
                                        original_result['body_part'].title(),
                                        original_result['arthroplasty'].title(),
                                        original_result['further_information_needed'].title()
                                    ]
                                })
                                st.dataframe(df, hide_index=True)
                                
                                # Add dropdowns for modification
                                st.markdown("### Modify Analysis")
                                modified_result = {}
                                
                                modified_result['procedure_type'] = st.selectbox(
                                    'Procedure Type',
                                    options=['arthroplasty', 'soft tissue', 'unknown'],
                                    index=['arthroplasty', 'soft tissue', 'unknown'].index(original_result['procedure_type'])
                                )
                                
                                modified_result['body_part'] = st.selectbox(
                                    'Body Part',
                                    options=['hip', 'knee', 'unknown'],
                                    index=['hip', 'knee', 'unknown'].index(original_result['body_part'])
                                )
                                
                                if modified_result['procedure_type'] == 'arthroplasty':
                                    modified_result['arthroplasty'] = st.selectbox(
                                        'Arthroplasty Type',
                                        options=['primary', 'revision', 'unknown'],
                                        index=['primary', 'revision', 'unknown'].index(
                                            original_result['arthroplasty'] if original_result['arthroplasty'] != 'N/A' else 'unknown'
                                        )
                                    )
                                else:
                                    modified_result['arthroplasty'] = 'N/A'
                                
                                modified_result['further_information_needed'] = st.selectbox(
                                    'Further Information Needed',
                                    options=['yes', 'no'],
                                    index=['yes', 'no'].index(original_result['further_information_needed'])
                                )
                            else:
                                st.error(f"Error in Llama analysis: {llama_result}")
                        else:
                            print("Using regular analysis path...")
                            # Use the values calculated in Decision Logic section
                            original_result = {
                                'procedure_type': procedure_type.lower(),
                                'body_part': primary_body_part.lower(),
                                'arthroplasty': 'N/A' if procedure_type.lower() != 'arthroplasty' else 'unknown',
                                'further_information_needed': 'yes' if pattern == "unclear pathology" else 'no'
                            }
                            
                            if original_result['body_part'] not in ['hip', 'knee']:
                                st.markdown("**This is not a hip nor knee referral letter**")
                            else:
                                # Create DataFrame and display
                                df = pd.DataFrame({
                                    'Field': ['Procedure Type', 'Body Part', 'Arthroplasty', 'Further Information Needed'],
                                    'Value': [
                                        original_result['procedure_type'].title(),
                                        original_result['body_part'].title(),
                                        original_result['arthroplasty'].title(),
                                        original_result['further_information_needed'].title()
                                    ]
                                })
                                st.dataframe(df, hide_index=True)
                                
                                # Add dropdowns for modification
                                st.markdown("### Modify Analysis")
                                modified_result = {}
                                
                                modified_result['procedure_type'] = st.selectbox(
                                    'Procedure Type',
                                    options=['arthroplasty', 'soft tissue', 'unknown'],
                                    index=['arthroplasty', 'soft tissue', 'unknown'].index(original_result['procedure_type'])
                                )
                                
                                modified_result['body_part'] = st.selectbox(
                                    'Body Part',
                                    options=['hip', 'knee', 'unknown'],
                                    index=['hip', 'knee', 'unknown'].index(original_result['body_part'])
                                )
                                
                                if modified_result['procedure_type'] == 'arthroplasty':
                                    modified_result['arthroplasty'] = st.selectbox(
                                        'Arthroplasty Type',
                                        options=['primary', 'revision', 'unknown'],
                                        index=['primary', 'revision', 'unknown'].index(
                                            original_result['arthroplasty'] if original_result['arthroplasty'] != 'N/A' else 'unknown'
                                        )
                                    )
                                else:
                                    modified_result['arthroplasty'] = 'N/A'
                                
                                modified_result['further_information_needed'] = st.selectbox(
                                    'Further Information Needed',
                                    options=['yes', 'no'],
                                    index=['yes', 'no'].index(original_result['further_information_needed'])
                                )
                                
                                # Save button
                                if st.button('Save Changes'):
                                    changes_saved = save_analysis_changes(
                                        original_result,
                                        modified_result,
                                        analysis_id,
                                        uploaded_file.name
                                    )
                                    if changes_saved:
                                        st.success('Changes saved successfully!')
                                    else:
                                        st.info('No changes detected.')
                        
                        print(f"Analysis complete. ID: {analysis_id}")
                        st.markdown(f"\nAnalysis ID: {analysis_id}")

                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")

    st.write("Upload PDF files to analyze medical referrals.")

if __name__ == "__main__":
    main()
