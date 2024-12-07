import math
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def load_inverted_index(file_path):
    """Load the inverted index."""
    word_data = defaultdict(dict)
    with open(file_path, "r") as file:
        for line in file:
            if not line.strip():
                continue
            word, occurrences = line.split(":", 1)
            occurrences = eval(occurrences.strip())
            for doc, positions in occurrences:
                word_data[word][doc] = len(positions)
    return word_data

def generate_document_texts(word_data):
    """Generate document representations."""
    doc_texts = defaultdict(list)
    for word, doc_data in word_data.items():
        for doc, freq in doc_data.items():
            doc_texts[doc].extend([word] * freq)
    document_texts = [" ".join(words) for words in doc_texts.values()]
    document_names = list(doc_texts.keys())
    return document_texts, document_names

def save_cluster_results(cluster_terms, cluster_file):
    """Save clustering results with top terms and TF-IDF weights."""
    with open(cluster_file, "w") as f:
        for cluster, terms in cluster_terms.items():
            f.write(f"Cluster {cluster + 1}:\n")
            for term, weight in terms[:20]:  # Top 20 terms
                f.write(f"{term}: {weight:.4f}\n")
            f.write("\n")

def cluster_documents(document_texts, n_clusters, output_file, top_n_terms=20):
    """Perform clustering and save results."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(document_texts)
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(tfidf_matrix)

    feature_names = vectorizer.get_feature_names_out()
    cluster_terms = {}

    for i in range(n_clusters):
        cluster_indices = (labels == i).nonzero()[0]
        cluster_tfidf = tfidf_matrix[cluster_indices].mean(axis=0).tolist()[0]
        top_term_indices = sorted(range(len(cluster_tfidf)), key=lambda x: cluster_tfidf[x], reverse=True)[:top_n_terms]
        cluster_terms[i] = [(feature_names[idx], cluster_tfidf[idx]) for idx in top_term_indices]

        print(f"\nTop {top_n_terms} Terms for Cluster {i + 1}:")
        for term, weight in cluster_terms[i]:
            print(f"{term}: {weight:.4f}")

    save_cluster_results(cluster_terms, output_file)
    return labels, cluster_terms

def plot_clusters(tfidf_matrix, labels, document_names, filename="clusters.png"):
    """Visualize clusters."""
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(tfidf_matrix.toarray())
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap="viridis")
    plt.title("Document Clustering")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.savefig(filename)
    plt.close()

def main():
    file_path = "inverted_index.txt"
    word_data = load_inverted_index(file_path)
    document_texts, document_names = generate_document_texts(word_data)

    # Clustering for 3 clusters
    n_clusters_3 = min(3, len(document_names))
    labels_3, top_terms_3 = cluster_documents(
        document_texts,
        n_clusters_3,
        "clustering_3_clusters.txt",
        top_n_terms=20
    )

    # Clustering for 6 clusters
    n_clusters_6 = min(6, len(document_names))
    labels_6, top_terms_6 = cluster_documents(
        document_texts,
        n_clusters_6,
        "clustering_6_clusters.txt",
        top_n_terms=20
    )

    # Visualize clusters
    tfidf_matrix = TfidfVectorizer().fit_transform(document_texts)
    plot_clusters(tfidf_matrix, labels_3, document_names, filename="clusters_3.png")
    plot_clusters(tfidf_matrix, labels_6, document_names, filename="clusters_6.png")

if __name__ == "__main__":
    main()
