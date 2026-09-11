# AI Resume Redactor

An AI-powered tool to automatically redact personal information from PDF resumes while preserving all formatting.

## Features

✅ **AI-Powered PII Detection** - Uses Hugging Face BERT model + Pattern matching  
✅ **Redacts**: Names, Emails, Phone numbers, Addresses, Date of Birth, ID numbers  
✅ **Preserves**: Education, Experience, Skills, Companies, Job titles  
✅ **Perfect Formatting** - Directly edits PDFs with black boxes, no format loss  
✅ **Safe Detection** - Validates entities to avoid false positives

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/alizeeshan-dev/HR-Resume-Screener.git
cd HR-Resume-Screener
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

**Note:** First run will download the AI model (~431MB) from Hugging Face.

## Usage

### Quick Start

```python
from resume_redactor_pdf import ResumeRedactorPDF

# Initialize the redactor (loads AI model)
redactor = ResumeRedactorPDF()

# Process a PDF resume
output_file, redacted_items = redactor.process_file("resume.pdf")

print(f"Redacted PDF saved: {output_file}")
print(f"Items redacted: {len(redacted_items)}")
```

### Output

- Input: `resume.pdf`
- Output: `resume_redacted.pdf` (same format, PII blacked out)

### What Gets Redacted

Personal information is blacked out in the PDF:

- **Names** - John Smith → **BLACK BOX**
- **Emails** - john.smith@gmail.com → **BLACK BOX**
- **Phone** - +1-555-123-4567 → **BLACK BOX**
- **Addresses** - San Francisco, CA → **BLACK BOX**
- **Date of Birth** - 11 MAY, 1988 → **BLACK BOX**
- **ID Numbers** - NIC/Passport numbers → **BLACK BOX**

### What Is Preserved

Professional information remains intact:

- Education (degrees, universities)
- Work experience (job titles, responsibilities)
- Skills (technical and soft skills)
- Companies worked at
- Graduation years

## Files

- `resume_redactor_pdf.py` - Main redaction class with full functionality
- `simple_pdf_redactor.py` - Simple usage example
- `requirements.txt` - Python dependencies

## How It Works

1. **Extract Text** - Reads PDF using PyMuPDF
2. **AI Detection** - Uses `yashpwr/resume-ner-bert-v2` BERT model to identify entities
3. **Pattern Matching** - Regex patterns catch emails, phones, DOB, addresses, ID numbers
4. **Validation** - Filters out false positives (min length checks)
5. **Redaction** - Directly blacks out detected PII in PDF using PyMuPDF
6. **Output** - Saves redacted PDF with perfect formatting preserved

## Example

Run the included example:

```python
python simple_pdf_redactor.py
```

## Dependencies

- PyPDF2 - PDF text extraction
- python-docx - DOCX support
- transformers - Hugging Face models
- torch - PyTorch for model inference
- sentencepiece - Tokenization
- protobuf - Model serialization
- pymupdf (fitz) - PDF editing and redaction

## Model Information

- **Model**: `yashpwr/resume-ner-bert-v2` from Hugging Face
- **Type**: BERT-based Named Entity Recognition
- **Trained for**: Resume entity extraction
- **Size**: ~431MB
