# 🏥 Orthopedic Referral Triage Assistant

An AI-powered tool that helps classify and prioritize orthopedic referrals for hip and knee conditions. The system uses OCR to extract text from PDF referrals and analyzes them using both rule-based and LLM-based approaches.

## 🌟 Features

- 📄 PDF Text Extraction with OCR
- 🤖 AI-powered referral analysis
- 🔍 Identification of:
  - Body part (hip/knee)
  - Procedure type (arthroplasty/soft tissue)
  - Arthroplasty type (primary/revision)
- 📊 Analysis accuracy tracking
- 💾 Historical data storage and analysis

## 🚀 Getting Started

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
For Ubuntu/Debian
sudo apt-get install tesseract-ocr
For Mac
brew install tesseract
For Windows
Download installer from: https://github.com/UB-Mannheim/tesseract/wiki

```

5. Install and start Ollama:

```bash
Follow instructions at: https://ollama.ai/download
Then pull the required model:
ollama pull llama3.1:70b
ollama run llama3.1:70b
```

### Running the Application

1. Activate the virtual environment:

```bash
source .venv/bin/activate
```

2. Start the Streamlit app:

```bash
streamlit run inference/deploy.py
```

3. Open the app in your browser:

```bash
http://localhost:8501
```


## 📁 Project Structure

```
.
├── inference/
│ └── deploy.py # Main application file
├── datasets/ # Created automatically for storing analysis data
├── requirements.txt # Project dependencies
└── README.md
```


## 💡 Usage

1. Upload one or more PDF referral letters through the web interface
2. The system will automatically:
   - Extract text using OCR
   - Analyze the content
   - Display results and confidence scores
3. You can:
   - Review the analysis
   - Modify classifications if needed
   - Request more information
   - Accept referrals
   - View historical statistics

## 📊 Data Storage

The application automatically creates and manages three CSV files in the `datasets` directory:
- `analysis_history.csv`: Records all analyses
- `analysis_changes.csv`: Tracks modifications to analyses
- `referral_actions.csv`: Stores referral actions (info requests/acceptances)

## ⚠️ Important Notes

- Ensure Tesseract OCR is properly installed and accessible
- Ollama must be running for LLM functionality
- The virtual environment must be activated before running the application
- Make sure you have sufficient disk space for storing analysis data