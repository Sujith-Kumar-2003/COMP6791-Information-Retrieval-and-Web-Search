import os
import fitz  # PyMuPDF
import pytesseract
from pytesseract import Output
from PIL import Image
from collections import defaultdict
import re
from urllib.parse import unquote  # Import to decode URL-encoded file names

# List of 150 common stopwords (this can be extended or customized)
STOPWORDS = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
    'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
    'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now',
    'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn',
    'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn',
    'wasn', 'weren', 'won', 'wouldn'
])

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
        """Tokenize the text into lowercase alphanumeric words and remove stopwords."""
        tokens = re.findall(r'\b\w+\b', text.lower())  # Tokenize and make lowercase
        return [token for token in tokens if token not in STOPWORDS]  # Remove stopwords

    def save_to_file(self, file_path):
        """Saves the inverted index to a file."""
        with open(file_path, 'w', encoding='utf-8') as f:
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
        # Decode the URL-encoded file name to get the correct name
        doc_id = unquote(os.path.splitext(os.path.basename(pdf_path))[0])  # Decode filename
        inverted_index.add_document(doc_id, text)
        print(f"Indexed: {doc_id}")
    else:
        print(f"Failed to extract any text from: {pdf_path}")

if __name__ == "__main__":
    # Root directory containing subdirectories of PDFs
    root_pdf_dir = "Downloaded_PDFs"
    output_file = "inverted_index.txt"

    inverted_index = InvertedIndex()

    # Iterate through all the directories and PDFs
    for root, _, files in os.walk(root_pdf_dir):
        for file in files:
            if file.endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                process_pdf(pdf_path, inverted_index)

    # Save the inverted index to a file
    inverted_index.save_to_file(output_file)
    print(f"Inverted index saved to {output_file}")
