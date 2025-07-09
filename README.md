# Orthopedic Referral Triage Assistant

An AI-powered tool that helps classify and prioritize orthopedic referrals for hip and knee conditions. The system uses OCR to extract text from PDF referrals and analyzes them using both rule-based and LLM-based approaches.

## Features

- PDF Text Extraction with OCR
- AI-powered referral analysis
- Identification of:
  - Body part (hip/knee)
  - Procedure type (arthroplasty/soft tissue)
  - Arthroplasty type (primary/revision)
- Analysis accuracy tracking
- Historical data storage and analysis

## Getting Started

### Prerequisites

- Python 3.8+
- Virtual environment
- Tesseract OCR
- Ollama (for LLM functionality)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/orthopedic-triage.git
cd orthopedic-triage
```

2. Create and activate virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate # For Linux/Mac
.\.venv\Scripts\activate # For Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Install Tesseract OCR:

```bash
# For Ubuntu/Debian
sudo apt-get install tesseract-ocr

# For Mac
brew install tesseract

# For Windows
# Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
```

5. Install and start Ollama:

```bash
# Follow instructions at: https://ollama.ai/download
# Then pull the required model:
ollama pull llama3.3:70b
ollama run llama3.3:70b
```

### Running the Application

1. Activate the virtual environment:

```bash
source .venv/bin/activate
```

2. Start the Streamlit app:

```bash
# Current version with full functionality
streamlit run deploy_v2.py

# Legacy version (may require additional setup)
streamlit run inference/deploy.py
```

3. Open the app in your browser:

```
http://localhost:8501
```

## Project Structure

```
├── inference/
│ └── deploy.py             # Legacy application file
├── deploy_v2.py            # Current main application file
├── datasets/               # Created automatically for storing analysis data
├── requirements.txt        # Project dependencies
└── README.md
```

## Usage

1. Upload one or more PDF referral letters through the web interface
2. The system will automatically:
   - Extract text using OCR
   - Analyze the content
   - Display results and confidence scores
3. You can:
   - Review the analysis
   - Modify classifications if needed

## Future Features

- Request more information from referring clinicians
- Accept referrals and integrate with scheduling systems
- View historical statistics and analytics dashboard

## Data Storage

The application automatically creates and manages three CSV files in the `datasets` directory:

- `analysis_history.csv`: Records all analyses
- `analysis_changes.csv`: Tracks modifications to analyses
- `referral_actions.csv`: Stores referral actions (info requests/acceptances)

## Deployment Options

### Current Version (`deploy_v2.py`)

- Full OCR and AI analysis capabilities with LLM integration
- Real-time processing of PDF referrals
- Automatic fallback to rule-based analysis if LLM unavailable
- NHS-styled professional UI with referral cards
- Comprehensive error handling and logging

### Legacy Version (`inference/deploy.py`)

- Original implementation
- May require additional setup and configuration
- Use current version unless specific legacy features needed

## Important Notes

- Ensure Tesseract OCR is properly installed and accessible
- For full AI functionality, Ollama must be running with the required LLM model (llama3.3:70b)
- The application automatically falls back to rule-based analysis if the LLM is unavailable
- The virtual environment must be activated before running the application
- Make sure you have sufficient disk space for storing analysis data

## System Requirements

- **Operating System**: Linux, macOS, or Windows
- **Python**: Version 3.8 or higher
- **Memory**: Minimum 8GB RAM (32GB+ recommended for llama3.3:70b model)
- **Storage**: At least 50GB free disk space (for model storage)
- **Network**: Internet connection required for initial setup and LLM model downloads
- **Hardware**: GPU with 16GB+ VRAM recommended for optimal performance