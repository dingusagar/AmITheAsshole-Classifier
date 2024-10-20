#!/usr/bin/env python3

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap
import matplotlib.pyplot as plt
import sys

def main(args):
    file = args[0]
    # load your data
    data = pd.read_csv(file)

    # combine relevant text fields (title, body, top comments)
    data['combined_text'] = data['title'] + " " + data['body'] + " " + " ".join([f" {data[f'top_comment_{i}']}" for i in range(1, 11)])

    # preprocess the text with tf-idf
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X = vectorizer.fit_transform(data['combined_text'])

    # try different num of clusters and calc silhouette scores
    silhouette_scores = []
    k_values =[2,4,6,8,10,15,20,40,60,80,100]

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print(f"for n_clusters = {k}, the silhouette score is {silhouette_avg:.3f}")

    # plot silhouette scores vs. number of clusters
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, silhouette_scores, marker='o')
    plt.xlabel("number of clusters (k)")
    plt.ylabel("silhouette score")
    plt.title("silhouette score vs number of clusters")
    plt.show()

    # choose optimal cluster number based on silhouette scores
    optimal_k = k_values[silhouette_scores.index(max(silhouette_scores))]

    # fit k-means with the optimal number of clusters
    kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42)
    data['cluster'] = kmeans_optimal.fit_predict(X)

    # use umap for dimensionality reduction and visualization
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    X_umap = umap_model.fit_transform(X.toarray())

    # plot the umap visualization with clusters
    plt.figure(figsize=(10, 6))
    plt.scatter(X_umap[:, 0], X_umap[:, 1], c=data['cluster'], cmap='viridis', s=50)
    plt.colorbar(label='cluster')
    plt.title(f"umap visualization with {optimal_k} clusters")
    plt.show()

    # analyze clusters
    print(data.groupby('cluster')['verdict'].value_counts())

if __name__ == '__main__':
    main(sys.argv[1:])
