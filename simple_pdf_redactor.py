"""
Simple example: How to use the PDF Resume Redactor
"""

from resume_redactor_pdf import ResumeRedactorPDF

# Initialize the redactor (loads AI model)
redactor = ResumeRedactorPDF()

# Process a PDF resume - returns updated PDF with redactions
output_file, redacted_items = redactor.process_file("Resume - Maryam Mukhtiar Qazi.pdf")

print(f"\n✓ Redacted PDF created: {output_file}")
print(f"✓ Redacted {len(redacted_items)} items")
