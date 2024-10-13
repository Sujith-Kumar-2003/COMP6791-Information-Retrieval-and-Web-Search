import os
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

def extract_text_from_sgm_folder(folder_path):
    """
    Extracts title and body text from all .sgm files in the specified folder.
    """
    text_data = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.sgm'):
            file_path = os.path.join(folder_path, file_name)
            text_data += extract_text_from_sgm(file_path)
    return text_data

def extract_text_from_sgm(file_path):
    """
    Extracts text from a single .sgm file.
    """
    with open(file_path, 'r', encoding='latin-1') as file:
        soup = BeautifulSoup(file, 'html.parser')
        articles = soup.find_all('reuters')
        text_data = []
        for article in articles:
            title = article.find('title')
            body = article.find('body')
            if title and body:
                title_text = title.get_text().strip()
                body_text = body.get_text().strip()
                text_data.append((title_text, body_text))
        return text_data

def tokenize_text(text):
    """
    Tokenizes and cleans text by removing stop words and non-alphabetic tokens.
    """
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha()]  # Remove non-alphabetic tokens
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    return tokens
