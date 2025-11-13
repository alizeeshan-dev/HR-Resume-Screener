# HR-Resume-Screener
An AI-powered tool to automatically redact personal information from resumes (PDF and DOCX files).

## Features

✅ **Automatic PII Detection** using Hugging Face NER model
✅ **Redacts**: Names, Email addresses, Phone numbers, Physical addresses
✅ **Preserves**: Education, Experience, Skills, Company names, Job titles
✅ **Format Preservation**: Outputs DOCX with formatting intact
✅ **Pattern Matching**: Fallback regex patterns for emails and phone numbers

## Installation

All dependencies are already installed in your virtual environment:

- PyPDF2
- python-docx
- transformers
- torch
- sentencepiece
- protobuf

## Usage

### Quick Start

```python
from resume_redactor import ResumeRedactor

# Initialize the redactor
redactor = ResumeRedactor()

# Process a resume file (PDF or DOCX)
redactor.process_file("resume.pdf")
```

This will create:

- `resume_redacted.docx` - Redacted version with formatting
- `resume_redacted.txt` - Plain text version

### What Gets Redacted

Personal information is replaced with `***`:

- **Names** - John Smith → \*\*\*
- **Emails** - john.smith@gmail.com → \*\*\*
- **Phone** - +1-555-123-4567 → \*\*\*
- **Location** - San Francisco, CA → \*\*\*

### What Is Preserved

Professional information remains intact:

- Education (degrees, universities)
- Work experience (job titles, responsibilities)
- Skills (technical and soft skills)
- Companies worked at
- Graduation years

## Files

- `resume_redactor.py` - Main redaction class
- `example_usage.py` - Usage examples
- `testhugface.py` - Original NER model testing

## How It Works

1. **Extract Text**: Reads PDF/DOCX and extracts text content
2. **AI Detection**: Uses BERT-based NER model to identify PII entities
3. **Pattern Matching**: Additional regex patterns catch emails/phones
4. **Redaction**: Replaces all detected PII with `***`
5. **Output**: Saves redacted version in DOCX and TXT formats

## Example

**Original:**

```
John Smith
Email: john.smith@gmail.com | Phone: +1-555-123-4567
Location: San Francisco, CA

Senior Software Engineer at Google
Skills: Python, JavaScript, Machine Learning
```

**Redacted:**

```
***
Email: *** | Phone: ***
Location: ***

Senior Software Engineer at Google
Skills: Python, JavaScript, Machine Learning
```

## Model Information

Uses `yashpwr/resume-ner-bert-v2` from Hugging Face

- Trained specifically for resume entity extraction
- Supports 12 entity types
- BERT-based architecture
