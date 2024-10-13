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
