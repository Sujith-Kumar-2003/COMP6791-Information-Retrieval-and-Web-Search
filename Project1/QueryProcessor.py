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
