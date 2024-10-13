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
