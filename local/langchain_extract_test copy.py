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
from llama_cpp import Llama
import json
import traceback
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
import re
import uuid

logging.basicConfig(level=logging.INFO)

# Initialize the OCR model
model = ocr_predictor(pretrained=True)

# Initialize the Llama model
llm = Llama.from_pretrained(
    repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
    filename="Llama-3.2-3B-Instruct-IQ3_M.gguf",
)

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
        result = MedicalReferral(
            procedure_type='unknown',
            body_part='unknown',
            arthroplasty='unknown',
            further_information_needed='no'
        )
        
        text = text.lower()
        
        # Determine body part
        if any(term in text for term in ['hip', 'acetabul', 'femoral head']):
            result.body_part = 'hip'
        elif any(term in text for term in ['knee', 'patella', 'tibia', 'femur']):
            result.body_part = 'knee'
        
        # Determine procedure type and arthroplasty status
        if 'arthroplasty' in text or 'replacement' in text:
            result.procedure_type = 'arthroplasty'
            
            previous_treatment_indicators = [
                'previous', 'prior', 'earlier', 'before', 'already had',
                'underwent', 'revision', 'redo', 'failed'
            ]
            
            if any(indicator in text for indicator in previous_treatment_indicators):
                result.arthroplasty = 'revision'
            else:
                result.arthroplasty = 'primary'
            
            if 'primary arthroplasty' in text:
                result.arthroplasty = 'primary'
            elif 'revision arthroplasty' in text:
                result.arthroplasty = 'revision'
        elif any(term in text for term in ['soft tissue', 'ligament', 'tendon', 'muscle', 'bursa', 
                                           'meniscus', 'meniscal tear', 'sport injury']):
            result.procedure_type = 'soft tissue'
        
        # Check if further information is needed
        info_needed_indicators = ['need more', 'additional info', 'clarify', 'unclear', 'missing']
        if any(indicator in text for indicator in info_needed_indicators):
            result.further_information_needed = 'yes'
        
        analysis = f"""
        1. Procedure type: {result.procedure_type.capitalize()}
        2. Body part: {result.body_part.capitalize()}
        3. Arthroplasty: {result.arthroplasty.capitalize()}
        4. Further information needed: {result.further_information_needed.capitalize()}
        """
        
        return analysis
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
        for i, chunk in enumerate(chunks[:10]):  # Analyze up to 10 chunks
            print(f"Analyzing chunk {i+1}/10 (ID: {analysis_id})")
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

            response = llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}]
            )

            result_json = response['choices'][0]['message']['content']
            print(f"Llama response for chunk {i+1} (ID: {analysis_id}): {result_json}")
            
            try:
                json_match = re.search(r'\{[^}]+\}', result_json)
                if json_match:
                    result = json.loads(json_match.group())
                    if all(key in result for key in ["procedure_type", "body_part", "arthroplasty", "further_information_needed"]):
                        results.append(result)
                        print(f"Valid result found in chunk {i+1} (ID: {analysis_id})")
                    else:
                        print(f"Incomplete JSON in chunk {i+1} (ID: {analysis_id})")
                else:
                    print(f"No valid JSON found in chunk {i+1} (ID: {analysis_id})")
            except json.JSONDecodeError:
                print(f"Failed to parse JSON for chunk {i+1} (ID: {analysis_id})")

        if not results:
            return f"Error: No valid results from Llama analysis (ID: {analysis_id})"

        # Combine results from all analyzed chunks
        combined_result = {
            "procedure_type": max(set(r['procedure_type'] for r in results), key=lambda x: [r['procedure_type'] for r in results].count(x)),
            "body_part": max(set(r['body_part'] for r in results), key=lambda x: [r['body_part'] for r in results].count(x)),
            "arthroplasty": max(set(r['arthroplasty'] for r in results), key=lambda x: [r['arthroplasty'] for r in results].count(x)),
            "further_information_needed": "yes" if any(r['further_information_needed'] == "yes" for r in results) else "no"
        }

        print(f"Combined result (ID: {analysis_id}): {combined_result}")

        return format_result(combined_result, analysis_id)

    except Exception as e:
        logging.error(f"An error occurred in analyze_medical_referral_with_llama (ID: {analysis_id}): {str(e)}")
        logging.error(traceback.format_exc())
        return f"Error: {str(e)} (ID: {analysis_id})"
    finally:
        end_time = time.time()
        print(f"Medical referral analysis with Llama completed in {end_time - start_time:.2f} seconds (ID: {analysis_id})")

def format_result(result, analysis_id):
    return f"""
    Analysis ID: {analysis_id}
    1. Procedure type: {result['procedure_type'].capitalize()}
    2. Body part: {result['body_part'].capitalize()}
    3. Arthroplasty: {result['arthroplasty'].capitalize()}
    4. Further information needed: {result['further_information_needed'].capitalize()}
    """

async def process_medical_referral(pdf_file):
    print(f"Processing file: {pdf_file.name}")
    start_time = time.time()
    
    try:
        # Extract text directly without creating another temp file
        text = await extract_text_from_pdf(pdf_file)
        
        if not text or text.startswith("Error:"):
            st.error("No text was extracted from the PDF.")
            return "Error: No text extracted from PDF"

        st.write(f"Extracted text (first 200 characters): {text[:200]}...")

        # Analyze the extracted text
        analysis = analyze_medical_referral(text)

        end_time = time.time()
        print(f"Total processing time: {end_time - start_time:.2f} seconds")
        return analysis
    except Exception as e:
        logging.error(f"Error processing medical referral: {str(e)}")
        return f"Error: {str(e)}"

# Streamlit UI
def main():
    st.title("Medical Referral Analyzer")
    
    # Clear any cached data
    st.cache_data.clear()

    use_llama = st.checkbox("Use Llama 3.2 for analysis")

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

                    st.write(f"Extracted text (first 500 characters): {text[:500]}...")
                    
                    if use_llama:
                        analysis = analyze_medical_referral_with_llama(text)
                    else:
                        analysis = analyze_medical_referral(text)
                    
                    st.write("Analysis:")
                    st.write(analysis)
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")

    st.write("Upload PDF files to analyze medical referrals.")

if __name__ == "__main__":
    main()
