import fitz  # PyMuPDF
from docx import Document
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import os
import re

class ResumeRedactorPDF:
    def __init__(self):
        """Initialize the redactor with the NER model"""
        print("Loading redaction model...")
        model_name = "yashpwr/resume-ner-bert-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        print("Model loaded successfully!\n")
        
        # Define which entities to redact (only personal information)
        self.redact_entities = ['Name', 'Email Address', 'Phone', 'Location']
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF file using PyMuPDF"""
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        return text
    
    def extract_text_from_docx(self, docx_path):
        """Extract text from DOCX file"""
        doc = Document(docx_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    
    def extract_entities(self, text):
        """Extract PII entities from text using NER model"""
        # Handle long text by splitting into chunks
        max_length = 512
        chunks = self._split_text(text, max_length)
        all_entities = []
        
        for chunk_text, offset in chunks:
            # Tokenize
            inputs = self.tokenizer(
                chunk_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True
            )
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=2)
            
            # Extract entities
            entities = self._extract_entities_from_predictions(
                predictions[0], 
                inputs['input_ids'][0],
                chunk_text,
                offset
            )
            all_entities.extend(entities)
        
        return all_entities
    
    def _split_text(self, text, max_length):
        """Split text into chunks that fit within max_length tokens"""
        lines = text.split('\n')
        chunks = []
        current_chunk = ""
        current_offset = 0
        
        for line in lines:
            test_chunk = current_chunk + line + "\n"
            if len(test_chunk) * 0.25 < max_length - 50:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append((current_chunk, current_offset))
                    current_offset += len(current_chunk)
                current_chunk = line + "\n"
        
        if current_chunk:
            chunks.append((current_chunk, current_offset))
        
        return chunks if chunks else [(text, 0)]
    
    def _extract_entities_from_predictions(self, predictions, input_ids, chunk_text, offset):
        """Extract entities from model predictions"""
        entities = []
        current_entity = None
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        
        for i, pred in enumerate(predictions):
            label = self.model.config.id2label[pred.item()]
            token = tokens[i]
            
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
                continue
            
            if label.startswith('B-'):
                if current_entity:
                    entities.append(current_entity)
                
                entity_type = label[2:]
                current_entity = {
                    'tokens': [token],
                    'label': entity_type,
                    'start_idx': i
                }
                
            elif label.startswith('I-') and current_entity:
                current_entity['tokens'].append(token)
                
            elif label == 'O':
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        if current_entity:
            entities.append(current_entity)
        
        # Reconstruct text for each entity
        for entity in entities:
            entity['text'] = self._reconstruct_text(entity['tokens'])
            del entity['tokens']
        
        return entities
    
    def _reconstruct_text(self, tokens):
        """Reconstruct text from BERT tokens"""
        text = ""
        for i, token in enumerate(tokens):
            if token.startswith("##"):
                text += token[2:]
            elif token in ['.', ',', '!', '?', ':', ';', ')', ']', '}', "'", '"', '-', '/', '@']:
                text += token
            elif i > 0 and tokens[i-1] in ['(', '[', '{', '@', '-', '/']:
                text += token
            else:
                if text and not text.endswith(' '):
                    text += " "
                text += token
        return text.strip()
    
    def find_pii_to_redact(self, text, entities):
        """Find all PII that needs to be redacted"""
        entities_to_redact = [e for e in entities if e['label'] in self.redact_entities]
        redact_list = []
        
        # Add NER-detected entities
        for entity in entities_to_redact:
            entity_text = entity['text']
            entity_text_cleaned = entity_text.replace(' .', '.').replace(' @', '@').replace(' -', '-')
            redact_list.append({
                'type': entity['label'],
                'text': entity_text_cleaned
            })
        
        # Pattern-based detection for fallback
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        for email in emails:
            if not any(email in item['text'] for item in redact_list):
                redact_list.append({
                    'type': 'Email Address (pattern)',
                    'text': email
                })
        
        # Phone pattern
        phone_patterns = [
            r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
            r'\b\d{10}\b'
        ]
        for pattern in phone_patterns:
            phones = re.findall(pattern, text)
            for phone in phones:
                if not any(phone in item['text'] for item in redact_list):
                    redact_list.append({
                        'type': 'Phone (pattern)',
                        'text': phone
                    })
        
        # Name pattern
        name_patterns = [
            r'Name\s+of\s+Expert\s*:\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,4})(?=\s*\n)',
            r'(?:^|\n)Name\s*:\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,4})(?=\s*\n)',
        ]
        for pattern in name_patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            for name in matches:
                name = ' '.join(name.split())
                if name and len(name) > 3 and not any(name in item['text'] for item in redact_list):
                    redact_list.append({
                        'type': 'Name (pattern)',
                        'text': name
                    })
        
        # Father's Name pattern (common in South Asian resumes)
        fathers_name_patterns = [
            r"Father'?s?\s+Name\s*:\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,4})(?=\s*\n)",
        ]
        for pattern in fathers_name_patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            for name in matches:
                name = ' '.join(name.split())
                if name and len(name) > 3 and not any(name in item['text'] for item in redact_list):
                    redact_list.append({
                        'type': "Father's Name (pattern)",
                        'text': name
                    })
        
        # Address pattern
        address_patterns = [
            r'Address\s*:\s+([A-Z0-9][^\n]{10,200}?)(?=\n\s*Date|$|\n\s*NIC|\n\s*Marital)',  # Stop before Date/NIC/Marital
        ]
        for pattern in address_patterns:
            matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
            for address in matches:
                address = address.strip()
                if address and not any(address in item['text'] for item in redact_list):
                    redact_list.append({
                        'type': 'Address (pattern)',
                        'text': address
                    })
        
        # Date of Birth pattern (multiple formats)
        dob_patterns = [
            r'Date\s+of\s+Birth\s*:\s+(\d{1,2}\s+[A-Z]+,?\s+\d{4})',  # 11 MAY, 1988
            r'Date\s+of\s+Birth\s*:\s+(\d{2}/\d{2}/\d{4})',  # 02/01/1993
            r'DOB\s*:\s+(\d{2}/\d{2}/\d{4})',
            r'Born\s*:\s+(\d{2}/\d{2}/\d{4})',
        ]
        for pattern in dob_patterns:
            matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
            for dob in matches:
                dob = dob.strip()
                if dob and not any(dob in item['text'] for item in redact_list):
                    redact_list.append({
                        'type': 'Date of Birth (pattern)',
                        'text': dob
                    })
        
        # NIC/ID/Passport Number pattern
        nic_patterns = [
            r'NIC\s+Number\s*:\s+([\d-]+)',
            r'National\s+ID\s*:\s+([\d-]+)',
            r'ID\s+Number\s*:\s+([\d-]+)',
            r'Passport\s*:\s+([A-Z0-9-]+)',
        ]
        for pattern in nic_patterns:
            matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
            for nic in matches:
                nic = nic.strip()
                if nic and len(nic) > 5 and not any(nic in item['text'] for item in redact_list):
                    redact_list.append({
                        'type': 'ID Number (pattern)',
                        'text': nic
                    })
        
        return redact_list
    
    def redact_pdf(self, input_pdf_path, output_pdf_path):
        """Redact PII from PDF and save to new file"""
        print(f"Processing: {input_pdf_path}")
        print("=" * 80)
        
        # Extract text
        text = self.extract_text_from_pdf(input_pdf_path)
        print(f"✓ Text extracted ({len(text)} characters)\n")
        
        # Extract PII entities
        print("Detecting personal information...")
        entities = self.extract_entities(text)
        print(f"✓ Found {len(entities)} entities\n")
        
        # Find all PII to redact
        print("Identifying items to redact...")
        redact_list = self.find_pii_to_redact(text, entities)
        print(f"✓ Identified {len(redact_list)} items to redact\n")
        
        # Show what will be redacted
        if redact_list:
            print("Items to redact:")
            print("-" * 80)
            for item in redact_list:
                print(f"  [{item['type']}]: {item['text']}")
            print()
        
        # Open PDF and redact
        print("Redacting PDF...")
        doc = fitz.open(input_pdf_path)
        redaction_count = 0
        
        for page in doc:
            for item in redact_list:
                # Search for text with variations in spacing
                text_to_find = item['text']
                
                # Try exact match first
                areas = page.search_for(text_to_find)
                
                # If no exact match, try with flexible spacing
                if not areas and ' ' in text_to_find:
                    # Try variations with different spacing
                    variations = [
                        text_to_find.replace(' ', '  '),  # Double space
                        text_to_find.replace(' ', '   '),  # Triple space
                    ]
                    for variation in variations:
                        areas = page.search_for(variation)
                        if areas:
                            break
                
                # Redact all found instances
                for rect in areas:
                    # Add redaction annotation
                    page.add_redact_annot(rect, fill=(0, 0, 0))  # Black fill
                    redaction_count += 1
            
            # Apply redactions on this page
            page.apply_redactions()
        
        # Save redacted PDF
        doc.save(output_pdf_path)
        doc.close()
        
        print(f"✓ Applied {redaction_count} redactions")
        print(f"✓ Redacted PDF saved to: {output_pdf_path}\n")
        
        print("=" * 80)
        print("Redaction complete!")
        print("=" * 80)
        
        return output_pdf_path, redact_list
    
    def process_file(self, file_path, output_path=None):
        """Main function to process and redact a resume file"""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if output_path is None:
            base_name = os.path.splitext(file_path)[0]
            output_path = f"{base_name}_redacted{file_ext}"
        
        if file_ext == '.pdf':
            return self.redact_pdf(file_path, output_path)
        elif file_ext in ['.docx', '.doc']:
            # For DOCX, we'll still use the text-based approach
            # (Direct DOCX editing is more complex)
            raise ValueError("DOCX redaction not yet implemented. Use PDF files.")
        else:
            raise ValueError(f"Unsupported file type: {file_ext}. Use PDF files.")


# Example usage
if __name__ == "__main__":
    # Initialize redactor
    redactor = ResumeRedactorPDF()
    
    # Test with Maryam's resume
    print("Testing PDF Redaction")
    print("\n")
    
    output_file, redacted_items = redactor.process_file(
        "Resume - Maryam Mukhtiar Qazi.pdf"
    )
    
    print(f"\nOutput saved to: {output_file}")
