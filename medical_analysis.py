"""Medical referral analysis functionality."""

import re
import json
import logging
import traceback
from config import (
    VALID_PROCEDURE_TYPES, VALID_BODY_PARTS, VALID_ARTHROPLASTY_TYPES,
    VALID_FURTHER_INFO, VALID_INJECTIONS, VALID_PHYSIOTHERAPY, AnalysisResult
)
from text_extraction import extract_patient_info, extract_xray_sentences

# Initialize the LLM
try:
    from langchain_ollama import OllamaLLM
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

def summarize_xray_findings_with_llm(xray_text: str) -> str:
    """Summarize X-ray findings using LLM"""
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
    """Analyze medical referral using LLM with fallback to rule-based analysis"""
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

def determine_priority(result):
    """Determine priority based on analysis result"""
    if result.get('arthroplasty_type') == 'revision':
        return 'Immediate'
    if result.get('procedure_type') == 'arthroplasty':
        return 'Urgent'
    return 'Routine'