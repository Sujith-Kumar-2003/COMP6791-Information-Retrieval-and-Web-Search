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
