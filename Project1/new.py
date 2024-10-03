from bs4 import BeautifulSoup
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer



def extract_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')

        # Extract articles and headlines
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


nltk.download('punkt')
nltk.download('stopwords')

def tokenize_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha()]  # Remove non-alphabetic tokens
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    return tokens

def spimi_index(token_stream):
    index = {}
    doc_id = 1  # Assuming you're tracking documents with ids

    for token in token_stream:
        if token in index:
            index[token].add(doc_id)
        else:
            index[token] = {doc_id}
    return index

def create_inverted_index(token_stream):
    inverted_index = {}
    for doc_id, tokens in enumerate(token_stream):
        for token in tokens:
            if token not in inverted_index:
                inverted_index[token] = []
            inverted_index[token].append(doc_id)
    return inverted_index

def create_positional_index(token_stream):
    positional_index = {}
    for doc_id, tokens in enumerate(token_stream):
        for pos, token in enumerate(tokens):
            if token not in positional_index:
                positional_index[token] = {}
            if doc_id not in positional_index[token]:
                positional_index[token][doc_id] = []
            positional_index[token][doc_id].append(pos)
    return positional_index

def apply_stemming(tokens):
    ps = PorterStemmer()
    return [ps.stem(token) for token in tokens]

def boolean_and(postings1, postings2):
    return postings1.intersection(postings2)

def boolean_or(postings1, postings2):
    return postings1.union(postings2)

def near_operator(term1, term2, k, positional_index):
    result_docs = []
    if term1 in positional_index and term2 in positional_index:
        for doc_id in positional_index[term1]:
            if doc_id in positional_index[term2]:
                positions1 = positional_index[term1][doc_id]
                positions2 = positional_index[term2][doc_id]
                for pos1 in positions1:
                    for pos2 in positions2:
                        if abs(pos1 - pos2) <= k:
                            result_docs.append(doc_id)
    return result_docs

def concordance(query, k, token_stream, positional_index):
    result = []
    if query in positional_index:
        for doc_id, positions in positional_index[query].items():
            for position in positions:
                left_context = token_stream[doc_id][max(0, position-k):position]
                right_context = token_stream[doc_id][position+1:min(len(token_stream[doc_id]), position+k+1)]
                result.append(f"{doc_id}: {' '.join(left_context)} {query} {' '.join(right_context)}")
    return result

# 1. `extract_text`: This function extracts the title and body from Reuters articles using BeautifulSoup for XML parsing.
# 2. `tokenize_text`: This function tokenizes text into words, removes non-alphabetic tokens and stopwords, making the text ready for indexing.
# 3. `spimi_index`: Implements a SPIMI-inspired indexer to create a simple inverted index where each token is mapped to document IDs.
# 4. `create_inverted_index`: Constructs a standard inverted index where each token points to a list of document IDs where it appears.
# 5. `create_positional_index`: Builds a positional index that keeps track of the position of each token within documents for more precise querying.
# 6. `apply_stemming`: This function reduces words to their stem (e.g., "running" to "run") to consolidate similar word forms during indexing.
# 7. `boolean_and` and `boolean_or`: These functions implement basic Boolean retrieval by performing intersection (AND) and union (OR) on document postings.
# 8. `near_operator`: This function retrieves documents where two search terms appear within a specified distance (k tokens) of each other.
# 9. `concordance`: Implements a concordance search, which returns a view of the corpus with the search term surrounded by its neighboring tokens, providing context for the occurrences of the query term.