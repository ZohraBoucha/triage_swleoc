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
        for chunk_num, chunk in enumerate(chunks[:10], 1):  # Use enumerate to track chunk number
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

            response = llm.invoke(prompt)  # Updated to use invoke instead of predict
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

def get_category_examples():
    return {
        "arthroplasty_hip": [
            """
            78-year-old lady with severe degenerative changes in her right hip. X-rays show joint space narrowing 
            and osteophyte formation. Patient reports groin pain, difficulty with walking and activities of daily living. 
            Failed conservative management including physiotherapy.
            """,
            """
            65-year-old male with advanced osteoarthritis of the left hip. MRI shows bone-on-bone changes.
            Patient has significant pain on weight bearing and limited range of motion. Pain affecting sleep and mobility.
            """,
            """
            55-year-old female with avascular necrosis of the right hip confirmed on imaging.
            Progressive pain over 6 months, now requiring walking stick. Limited hip rotation and groin pain.
            """
        ],
        
        "soft_tissue_hip": [
            """
            32-year-old athlete with right hip pain. MRI shows labral tear.
            Pain worse with flexion and rotation. No degenerative changes noted.
            Clicking sensation with movement. Failed physiotherapy.
            """,
            """
            45-year-old with hip impingement syndrome. Pain on hip flexion and internal rotation.
            Positive impingement test. X-rays show cam deformity but preserved joint space.
            """,
            """
            28-year-old runner with lateral hip pain. Clinical features of trochanteric bursitis.
            Pain on palpation of greater trochanter. Normal hip joint x-rays.
            """
        ],
        
        "arthroplasty_knee": [
            """
            72-year-old with end-stage osteoarthritis of right knee. X-rays show tricompartmental changes
            with bone-on-bone in medial compartment. Constant pain, difficulty with stairs, failed injections.
            """,
            """
            68-year-old with severe bilateral knee arthritis, worse on right. Significant varus deformity.
            X-rays show complete loss of joint space medially. Unable to walk more than 100 yards.
            """,
            """
            75-year-old with post-traumatic arthritis of left knee. Previous tibial plateau fracture.
            Now shows advanced degenerative changes. Persistent pain despite conservative measures.
            """
        ],
        
        "soft_tissue_knee": [
            """
            25-year-old footballer with acute knee injury. MRI confirms ACL rupture.
            Positive Lachman test. No degenerative changes seen on x-ray.
            Episodes of giving way.
            """,
            """
            50-year-old with medial knee pain. MRI shows complex medial meniscal tear.
            Mechanical symptoms including locking. No significant arthritis on x-rays.
            """,
            """
            35-year-old with anterior knee pain. Clinical features of patellofemoral syndrome.
            Pain worse on stairs. Normal x-rays. Failed physiotherapy regime.
            """
        ]
    }

def analyze_medical_referral_with_examples(text, examples):
    # Compare the input text with examples from each category
    similarities = {}
    for category, example_list in examples.items():
        for example in example_list:
            # Calculate similarity between input text and example
            # (You could use various methods here - simple term matching,
            # embeddings, or more sophisticated NLP techniques)
            similarity_score = calculate_similarity(text, example)
            similarities[category] = max(similarities.get(category, 0), similarity_score)
    
    # Determine the most likely category
    most_likely_category = max(similarities.items(), key=lambda x: x[1])[0]
    
    # Extract procedure type and body part from category
    procedure_type, body_part = most_likely_category.split('_')
    
    return {
        "procedure_type": procedure_type.title(),
        "body_part": body_part.title(),
        "confidence_score": similarities[most_likely_category]
    }

# Streamlit UI
def main():
    st.title("Medical Referral Analyzer")
    
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
                                'despite physiotherapy', 'in agony'
                            ]
                            
                            if any(indicator in text_lower for indicator in arthroplasty_indicators):
                                procedure_type = "Arthroplasty"
                                # Find the matching indicator for evidence
                                matching_indicator = next((indicator for indicator in arthroplasty_indicators if indicator in text_lower), None)
                                if matching_indicator:
                                    start = max(0, text_lower.find(matching_indicator) - 50)
                                    end = min(len(text), text_lower.find(matching_indicator) + len(matching_indicator) + 50)
                                    clinical_evidence = text[start:end].strip()
                            
                            st.markdown(f"- **Procedure Type:** {procedure_type}")
                            if clinical_evidence:
                                st.markdown(f"  - Evidence: '{clinical_evidence}'")
                            
                            # Determine pattern and intervention
                            if procedure_type == "Arthroplasty":
                                pattern = "degenerative joint disease"
                                intervention = "arthroplasty"
                            else:
                                pattern = "unclear pathology"
                                intervention = "further investigation needed"
                            
                            st.markdown("Based on the clinical picture:")
                            st.markdown(f"- Pattern suggests {pattern}")
                            
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
                            
                            # Create results data from Llama analysis
                            results = {
                                'Field': ['Procedure Type', 'Body Part'],
                                'Value': [llama_result['procedure_type'], llama_result['body_part']]
                            }
                            
                            if llama_result['procedure_type'] == "Arthroplasty":
                                results['Field'].append('Arthroplasty')
                                results['Value'].append(llama_result['arthroplasty'])
                            
                            results['Field'].append('Further Information Needed')
                            results['Value'].append(llama_result['further_information_needed'])
                        else:
                            print("Using regular analysis path...")
                            # Get the values from the previous analysis
                            body_part = primary_body_part  # This was set in Decision Logic
                            procedure_type = procedure_type  # This was set in Decision Logic
                            
                            # Create results data from the analysis
                            analysis_result = analyze_medical_referral(text)
                            
                            if analysis_result.body_part not in ['hip', 'knee']:
                                st.markdown("**This is not a hip nor knee referral letter**")
                            else:
                                results = {
                                    'Field': ['Procedure Type', 'Body Part'],
                                    'Value': [analysis_result.procedure_type.title(), analysis_result.body_part.title()]
                                }
                                
                                if analysis_result.procedure_type == 'arthroplasty':
                                    results['Field'].append('Arthroplasty')
                                    results['Value'].append(analysis_result.arthroplasty.title())
                                
                                results['Field'].append('Further Information Needed')
                                results['Value'].append(analysis_result.further_information_needed.title())
                                
                                # Create DataFrame and display
                                df = pd.DataFrame(results)
                                st.dataframe(df, hide_index=True)
                        
                        print(f"Analysis complete. ID: {analysis_id}")
                        st.markdown(f"\nAnalysis ID: {analysis_id}")

                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")

    st.write("Upload PDF files to analyze medical referrals.")

if __name__ == "__main__":
    main()
