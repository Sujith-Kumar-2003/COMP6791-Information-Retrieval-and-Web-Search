from PreprocessingPipeline import extract_text_from_sgm_folder, tokenize_text
from PrimaryIndex import create_inverted_index
from PositionalIndex import create_positional_index
from QueryProcessor import boolean_and, boolean_or
from NEARoperator import near_operator
from CONCORDANCEoperator import concordance

def main():
    folder_path = '/Users/sujithkumaravel/Downloads/reuters21578'  # Update to your Reuters21578 dataset path
    print("Extracting text from Reuters21578 dataset...")
    text_data = extract_text_from_sgm_folder(folder_path)

    print("Tokenizing text...")
    token_stream = [tokenize_text(title + ' ' + body) for title, body in text_data]

    print("Building inverted index...")
    inverted_index = create_inverted_index(token_stream)

    print("Building positional index...")
    positional_index = create_positional_index(token_stream)

    term1 = "bush"
    term2 = "reagan"
    k = 5

    docs_and = boolean_and(set(inverted_index.get(term1, [])), set(inverted_index.get(term2, [])))
    docs_or = boolean_or(set(inverted_index.get(term1, [])), set(inverted_index.get(term2, [])))
    docs_near = near_operator(term1, term2, k, positional_index)
    concordance_result = concordance("climate", 10, token_stream, positional_index)

    print(f"Documents containing both terms '{term1}' and '{term2}': {docs_and}")
    print(f"Documents containing either term '{term1}' or '{term2}': {docs_or}")
    print(f"Documents with terms '{term1}' and '{term2}' within {k} tokens: {docs_near}")
    print(f"Concordance for term 'climate':")
    for line in concordance_result:
        print(line)

if __name__ == "__main__":
    main()
