import os
import fitz  # PyMuPDF
import pytesseract
from pytesseract import Output
from PIL import Image
from collections import defaultdict
import re


# Try to change the indexing like directly read the content of the pdf and directly index it, rather than down alod eth file to the PC
class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(list)

    def add_document(self, doc_id, text):
        """Tokenizes the text and adds it to the inverted index."""
        tokens = self.tokenize(text)
        token_positions = defaultdict(list)

        for position, token in enumerate(tokens):
            token_positions[token].append(position)

        for token, positions in token_positions.items():
            self.index[token].append((doc_id, positions))

    @staticmethod
    def tokenize(text):
        """Tokenize the text into lowercase alphanumeric words."""
        return re.findall(r'\b\w+\b', text.lower())

    def save_to_file(self, file_path):
        """Saves the inverted index to a file."""
        with open(file_path, 'w') as f:
            for token, occurrences in self.index.items():
                f.write(f"{token}: {occurrences}\n")

    def __str__(self):
        """String representation of the inverted index."""
        return "\n".join(f"{token}: {occurrences}" for token, occurrences in self.index.items())

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF using PyMuPDF."""
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        if not text.strip():
            raise ValueError("PDF contains no selectable text.")
    except Exception as e:
        print(f"Failed to extract text using PyMuPDF: {e}")
        text = None
    return text

def extract_text_with_ocr(pdf_path):
    """Extract text from a PDF using OCR."""
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page_number in range(len(doc)):
            pix = doc[page_number].get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            ocr_text = pytesseract.image_to_string(img, lang='eng', config='--psm 6')
            text += ocr_text
    except Exception as e:
        print(f"Error during OCR processing: {e}")
    return text

def process_pdf(pdf_path, inverted_index):
    """Process a PDF to extract text and add it to the inverted index."""
    text = extract_text_from_pdf(pdf_path)
    if not text:
        print(f"Falling back to OCR for: {pdf_path}")
        text = extract_text_with_ocr(pdf_path)

    if text.strip():
        inverted_index.add_document(pdf_path, text)
        print(f"Indexed: {pdf_path}")
    else:
        print(f"Failed to extract any text from: {pdf_path}")

if __name__ == "__main__":
    # Directory containing downloaded PDFs
    pdf_dir = "Downloaded_PDFs"
    output_file = "inverted_index.txt"

    inverted_index = InvertedIndex()

    # Iterate through PDFs in the directory
    for root, _, files in os.walk(pdf_dir):
        for file in files:
            if file.endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                process_pdf(pdf_path, inverted_index)

    # Save the inverted index to a file
    inverted_index.save_to_file(output_file)
    print(f"Inverted index saved to {output_file}")
