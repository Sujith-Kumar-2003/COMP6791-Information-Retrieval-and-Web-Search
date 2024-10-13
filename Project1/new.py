import os
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd

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

def tokenize_text(text):
    """
    Tokenizes and cleans text by removing stop words and non-alphabetic tokens.
    """
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha()]  # Remove non-alphabetic tokens
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    return tokens

# Example of extracting text from the Reuters21578 dataset
folder_path = '/Users/sujithkumaravel/Downloads/reuters21578'
text_data = extract_text_from_sgm_folder(folder_path)

# Tokenize the extracted text data
token_stream = [tokenize_text(title + ' ' + body) for title, body in text_data]

def spimi_index(token_stream):
    """
    Creates a simple SPIMI-inspired inverted index where tokens map to document IDs.
    """
    index = {}
    doc_id = 0  # Use doc_id to track documents

    for tokens in token_stream:
        doc_id += 1
        for token in tokens:
            if token in index:
                index[token].add(doc_id)
            else:
                index[token] = {doc_id}
    return index

def create_inverted_index(token_stream):
    """
    Creates an inverted index mapping each token to the list of document IDs where it appears.
    """
    inverted_index = {}
    for doc_id, tokens in enumerate(token_stream):
        for token in tokens:
            if token not in inverted_index:
                inverted_index[token] = []
            inverted_index[token].append(doc_id)
    return inverted_index

def create_positional_index(token_stream):
    """
    Creates a positional inverted index where tokens are mapped to their positions within documents.
    """
    positional_index = {}
    for doc_id, tokens in enumerate(token_stream):
        for pos, token in enumerate(tokens):
            if token not in positional_index:
                positional_index[token] = {}
            if doc_id not in positional_index[token]:
                positional_index[token][doc_id] = []
            positional_index[token][doc_id].append(pos)
    return positional_index

# Build inverted index and positional index
inverted_index = create_inverted_index(token_stream)
positional_index = create_positional_index(token_stream)

def boolean_and(postings1, postings2):
    """
    Performs an AND operation between two sets of postings (intersection).
    """
    return postings1.intersection(postings2)

def boolean_or(postings1, postings2):
    """
    Performs an OR operation between two sets of postings (union).
    """
    return postings1.union(postings2)

def near_operator(term1, term2, k, positional_index):
    """
    Implements the NEAR operator to find documents where term1 and term2 appear within k tokens of each other.
    """
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
    """
    Implements the CONCORDANCE function which returns occurrences of the query term with k tokens of context.
    """
    result = []
    if query in positional_index:
        for doc_id, positions in positional_index[query].items():
            for position in positions:
                left_context = token_stream[doc_id][max(0, position-k):position]
                right_context = token_stream[doc_id][position+1:min(len(token_stream[doc_id]), position+k+1)]
                result.append(f"{doc_id}: {' '.join(left_context)} {query} {' '.join(right_context)}")
    return result

def calculate_statistics(token_stream):
    """
    Calculates statistical properties of terms in the token stream and returns a DataFrame.
    """
    term_stats = []
    total_tokens = 0

    for tokens in token_stream:
        total_tokens += len(tokens)

    # Count terms and postings at various preprocessing steps
    num_unfiltered_terms = len(set(token for tokens in token_stream for token in tokens))
    num_nonpositional_postings = total_tokens

    # Simulate preprocessing steps
    for step in [('No Numbers', 'no_numbers'),
                 ('Case Folding', 'case_folding'),
                 ('30 Stop Words', '30_stop_words'),
                 ('150 Stop Words', '150_stop_words'),
                 ('Stemming', 'stemming')]:

        step_name, step_method = step
        if step_method == 'no_numbers':
            filtered_tokens = [t for t in tokens if t.isalpha()]
        elif step_method == 'case_folding':
            filtered_tokens = [t.lower() for t in tokens if t.isalpha()]
        elif step_method == '30_stop_words':
            filtered_tokens = [t for t in tokens if t.isalpha() and t not in set(stopwords.words('english'))][:30]
        elif step_method == '150_stop_words':
            filtered_tokens = [t for t in tokens if t.isalpha() and t not in set(stopwords.words('english'))][:150]
        elif step_method == 'stemming':
            from nltk.stem import PorterStemmer
            ps = PorterStemmer()
            filtered_tokens = [ps.stem(t) for t in tokens if t.isalpha()]

        num_terms = len(set(filtered_tokens))
        num_postings = len(filtered_tokens)

        delta_percent_terms = ((num_unfiltered_terms - num_terms) / num_unfiltered_terms) * 100 if num_unfiltered_terms > 0 else 0
        delta_percent_postings = ((num_nonpositional_postings - num_postings) / num_nonpositional_postings) * 100 if num_nonpositional_postings > 0 else 0

        term_stats.append((step_name, num_terms, delta_percent_terms, num_postings, delta_percent_postings))

    # Convert to DataFrame for easy visualization
    df_stats = pd.DataFrame(term_stats, columns=['Preprocessing Step', 'Num Terms', '∆% Terms', 'Num Nonpositional Postings', '∆% Postings'])
    df_stats['T% Terms'] = ((df_stats['Num Terms'].cumsum() / num_unfiltered_terms) * 100).round(2)
    df_stats['T% Postings'] = ((df_stats['Num Nonpositional Postings'].cumsum() / num_nonpositional_postings) * 100).round(2)

    return df_stats

# Example queries
term1 = "bush"
term2 = "reagan"
k = 5

# Example Boolean AND, OR, and NEAR queries
docs_and = boolean_and(set(inverted_index.get(term1, [])), set(inverted_index.get(term2, [])))
docs_or = boolean_or(set(inverted_index.get(term1, [])), set(inverted_index.get(term2, [])))
docs_near = near_operator(term1, term2, k, positional_index)

# Example Concordance
concordance_result = concordance("climate", 10, token_stream, positional_index)

def main():
    # Phase 0: Extract and tokenize text from Reuters21578 dataset
    folder_path = '/Users/sujithkumaravel/Downloads/reuters21578'  # Update to your Reuters21578 dataset path
    print("Extracting text from Reuters21578 dataset...")
    text_data = extract_text_from_sgm_folder(folder_path)

    print("Tokenizing text...")
    token_stream = [tokenize_text(title + ' ' + body) for title, body in text_data]

    # Phase 1: Build the indexes
    print("Building inverted index...")
    inverted_index = create_inverted_index(token_stream)

    print("Building positional index...")
    positional_index = create_positional_index(token_stream)

    # Phase 2: Calculate statistical properties
    print("Calculating statistical properties...")
    df_stats = calculate_statistics(token_stream)
    print(df_stats)

    print(f"Documents containing both terms '{term1}' and '{term2}': {docs_and}")
    print(f"Documents containing either term '{term1}' or '{term2}': {docs_or}")
    print(f"Documents with terms '{term1}' and '{term2}' within {k} tokens: {docs_near}")
    print(f"Concordance for term 'climate':")
    for line in concordance_result:
        print(line)

if __name__ == "__main__":
    main()