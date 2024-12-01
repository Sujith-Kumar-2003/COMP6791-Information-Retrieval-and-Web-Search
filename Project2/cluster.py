import os
import fitz  # PyMuPDF for PDF reading
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Function to load text from PDFs in a folder
def load_pdf_text(pdf_folder):
    documents = []
    filenames = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith('.pdf') and not filename.startswith('~$'):  # Skip temporary files
            filepath = os.path.join(pdf_folder, filename)
            try:
                doc = fitz.open(filepath)
                text = ""
                for page in doc:
                    text += page.get_text()
                documents.append(text)
                filenames.append(filename)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    return documents, filenames

# Load PDF documents
pdf_folder = 'Downloaded_PDFs'  # Replace with your actual path
documents, filenames = load_pdf_text(pdf_folder)

# Vectorizing the documents using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)  # You can change max_features
X = vectorizer.fit_transform(documents)

# Apply KMeans clustering
n_clusters = 5  # You can change this value based on your needs
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X)

# Get the cluster centers (in high-dimensional space)
centers = kmeans.cluster_centers_

# Apply PCA to reduce the dimensionality to 2D for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.toarray())  # Transform the documents to 2D space

# Project the cluster centers into the 2D PCA space
center_2d = pca.transform(centers)

# Plotting the clusters and centroids
plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, cmap='viridis', marker='o', s=50)  # Plot documents
plt.scatter(center_2d[:, 0], center_2d[:, 1], c='red', marker='x', s=200)  # Plot centroids
plt.title('KMeans Clustering with PCA Projection')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.colorbar()
plt.grid(True)

# Save the plot to graph.png
plt.savefig('graph.png')

# Show the plot
plt.show()

# Print the top 50 most common words in each cluster based on TF-IDF weights
# Get the feature names (words)
feature_names = np.array(vectorizer.get_feature_names_out())

# Loop over each cluster and print the top 50 words along with their TF-IDF weights
for i in range(n_clusters):
    print(f"\nTop 50 words in cluster {i} with their TF-IDF weights:")
    # Get the indices of the words sorted by their TF-IDF weights for the current cluster center
    order_centroids = centers[i].argsort()[::-1]  # Sort the TF-IDF weights in descending order
    for ind in order_centroids[:50]:  # Loop over the top 50 words
        word = feature_names[ind]
        weight = centers[i][ind]  # Get the TF-IDF weight for the word
        print(f"{word}: {weight:.4f}")  # Print word and its corresponding TF-IDF weight
