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
