"""Streamlit UI components for the referral management system."""

import streamlit as st
import base64
import time
import asyncio
import uuid
import logging
import traceback
from config import (
    VALID_PROCEDURE_TYPES, VALID_BODY_PARTS, VALID_ARTHROPLASTY_TYPES,
    VALID_INJECTIONS, VALID_PHYSIOTHERAPY, VALID_FURTHER_INFO
)
from text_extraction import extract_text_from_pdf, extract_patient_info, extract_xray_sentences
from medical_analysis import analyze_medical_referral_with_llama, determine_priority, summarize_xray_findings_with_llm
from data_management import save_analysis_changes, save_analysis_history

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

def setup_page_config():
    """Setup Streamlit page configuration and styles"""
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

def display_header():
    """Display the application header with logo"""
    logo_path = "assets/swleoc_logo.png"
    if logo_base64 := get_image_as_base64(logo_path):
        st.markdown(f'<div class="app-header"><img src="data:image/png;base64,{logo_base64}" class="header-logo"><h1 class="header-title">Referral Management</h1></div>', unsafe_allow_html=True)
    else:
        st.title("🏥 Referral Management")

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