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

import logging
import streamlit as st
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# Import our modular components
from data_management import ensure_data_directories
from ui_components import (
    setup_page_config, display_header, handle_referrals_tab
)

logging.basicConfig(level=logging.INFO)

# Initialize the OCR model
try:
    model = ocr_predictor(pretrained=True)
except Exception as e:
    logging.warning(f"Could not initialize OCR model: {e}")
    model = None

def main():
    """Main Streamlit application"""
    ensure_data_directories()
    setup_page_config()
    display_header()

    st.cache_data.clear()
    tab1, tab2, tab3 = st.tabs(["Active Referrals", "Analytics Dashboard", "System Configuration"])
    
    with tab1:
        handle_referrals_tab()
    with tab2:
        st.header("Analytics Placeholder")
    with tab3:
        display_consultant_management()

if __name__ == "__main__":
    main()