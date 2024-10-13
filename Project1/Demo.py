# Import necessary libraries
from PreprocessingPipeline import extract_text_from_sgm_folder, tokenize_text
from PrimaryIndex import create_inverted_index
from PositionalIndex import create_positional_index
from QueryProcessor import boolean_and, boolean_or
from NEARoperator import near_operator
from CONCORDANCEoperator import concordance
import pandas as pd
from nltk.corpus import stopwords  # Ensure NLTK stopwords are available
from nltk.stem import PorterStemmer  # Import the PorterStemmer

def calculate_statistics(token_stream):
    """
    Calculates statistical properties of terms in the token stream and returns a DataFrame.
    """
    term_stats = []
    total_tokens = 0

    # Calculate total tokens
    for tokens in token_stream:
        total_tokens += len(tokens)

    # Count terms and postings
    num_unfiltered_terms = len(set(token for tokens in token_stream for token in tokens))
    num_nonpositional_postings = total_tokens

    # Simulate preprocessing steps and collect statistics
    for step in [('No Numbers', 'no_numbers'),
                 ('Case Folding', 'case_folding'),
                 ('30 Stop Words', '30_stop_words'),
                 ('150 Stop Words', '150_stop_words'),
                 ('Stemming', 'stemming')]:
        step_name, step_method = step
        filtered_tokens = []

        # Process according to the step method
        if step_method == 'no_numbers':
            filtered_tokens = [t for tokens in token_stream for t in tokens if t.isalpha()]
        elif step_method == 'case_folding':
            filtered_tokens = [t.lower() for tokens in token_stream for t in tokens if t.isalpha()]
        elif step_method == '30_stop_words':
            stop_words = set(stopwords.words('english'))
            filtered_tokens = [t for tokens in token_stream for t in tokens if t.isalpha() and t not in stop_words][:30]
        elif step_method == '150_stop_words':
            stop_words = set(stopwords.words('english'))
            filtered_tokens = [t for tokens in token_stream for t in tokens if t.isalpha() and t not in stop_words][:150]
        elif step_method == 'stemming':
            ps = PorterStemmer()
            filtered_tokens = [ps.stem(t) for tokens in token_stream for t in tokens if t.isalpha()]

        num_terms = len(set(filtered_tokens))
        num_postings = len(filtered_tokens)

        # Calculate percentage differences
        delta_percent_terms = ((num_unfiltered_terms - num_terms) / num_unfiltered_terms) * 100 if num_unfiltered_terms > 0 else 0
        delta_percent_postings = ((num_nonpositional_postings - num_postings) / num_nonpositional_postings) * 100 if num_nonpositional_postings > 0 else 0

        term_stats.append((step_name, num_terms, delta_percent_terms, num_postings, delta_percent_postings))

    # Convert to DataFrame for easy visualization
    df_stats = pd.DataFrame(term_stats, columns=['Preprocessing Step', 'Num Terms', '∆% Terms', 'Num Nonpositional Postings', '∆% Postings'])
    df_stats['T% Terms'] = ((df_stats['Num Terms'].cumsum() / num_unfiltered_terms) * 100).round(2)
    df_stats['T% Postings'] = ((df_stats['Num Nonpositional Postings'].cumsum() / num_nonpositional_postings) * 100).round(2)

    return df_stats

def main():
    folder_path = '/Users/sujithkumaravel/Downloads/reuters21578'
    print("Extracting text from Reuters21578 dataset...")
    text_data = extract_text_from_sgm_folder(folder_path)

    print("Tokenizing text...")
    token_stream = [tokenize_text(title + ' ' + body) for title, body in text_data]

    print("Building inverted index...")
    inverted_index = create_inverted_index(token_stream)

    print("Building positional index...")
    positional_index = create_positional_index(token_stream)

    # Query Processing
    term1 = "bush"
    term2 = "reagan"
    k = 5

    docs_and = boolean_and(set(inverted_index.get(term1, [])), set(inverted_index.get(term2, [])))
    docs_or = boolean_or(set(inverted_index.get(term1, [])), set(inverted_index.get(term2, [])))
    docs_near = near_operator(term1, term2, k, positional_index)
    concordance_result = concordance("climate", 10, token_stream, positional_index)

    # Display results
    print(f"Documents containing both terms '{term1}' and '{term2}': {docs_and}")
    print(f"Documents containing either term '{term1}' or '{term2}': {docs_or}")
    print(f"Documents with terms '{term1}' and '{term2}' within {k} tokens: {docs_near}")
    print(f"Concordance for term 'climate':")
    for line in concordance_result:
        print(line)

    # Calculate and display statistics
    statistics_df = calculate_statistics(token_stream)
    print("Statistics after preprocessing:")
    print(statistics_df)

if __name__ == "__main__":
    main()
