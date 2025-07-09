"""Data management and persistence functionality."""

import os
import logging
import pandas as pd
from datetime import datetime

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