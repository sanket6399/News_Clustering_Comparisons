import networkx as nx
import numpy as np
import igraph as ig
import leidenalg
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import csv

def load_and_preprocess_data(file_path):

    '''This function loads and preprocesses data to be used for clustering'''

    data = pd.read_csv(file_path)
    data['text'] = data['Headline'] + " " + data['Description'] + " " + data['Article text']
    return data

def create_networkx_graph(cosine_sim, threshold):

    '''Create NetworkX graph from cosine similarity matrix'''

    G = nx.Graph()
    # Add nodes
    for i in range(len(cosine_sim)):
        for j in range(i + 1, len(cosine_sim)):
            # the below code is to remove self similarity and similarity below threshold
            if cosine_sim[i][j] > 0.2:
                G.add_edge(i, j, weight=cosine_sim[i][j])
    return G

def convert_to_igraph(G):

    ''' This function converts NetworkX graph to iGraph graph'''

    nx_matrix = nx.to_numpy_array(G)
    # Get the row and column indices
    sources, targets = np.where(nx_matrix != 0)
    weights = nx_matrix[sources, targets]
    # Create the iGraph graph for Leiden algorithm
    igraph_graph = ig.Graph(edges=list(zip(sources, targets)), directed=False)
    igraph_graph.es['weight'] = weights
    return igraph_graph

def apply_leiden_algorithm(igraph_graph):

    '''This function applies Leiden algorithm to the iGraph graph'''

    start_time = time.time()
    # Apply Leiden algorithm
    partition = leidenalg.find_partition(igraph_graph, leidenalg.ModularityVertexPartition, weights='weight')
    end_time = time.time()
    # Calculate time taken for Leiden algorithm
    leiden_time = -1 * (start_time - end_time)
    print("Leiden time", leiden_time)
    return partition, leiden_time

def plot_clusters(clusters, G):

    '''This function plots the clusters'''

    i = 0
    for cluster_id, nodes in clusters.items():
        subgraph = G.subgraph(nodes)
        pos_subgraph = nx.spring_layout(subgraph)

        plt.figure(figsize=(8, 6))
        nx.draw(subgraph, pos_subgraph, node_color='yellow', with_labels=True, font_size=8, node_size=280)

        labels_subgraph = {node: str(node) for node in subgraph.nodes}
        nx.draw_networkx_labels(subgraph, pos_subgraph, labels_subgraph, font_size=8, font_color='black')

        plt.title(f"Cluster {i}")
        i += 1
        plt.show()

def base_function(file_path, threshold):

    '''This is the base function that calls all other functions'''

    # Load and preprocess data
    data = load_and_preprocess_data(file_path)

    # TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['text'])

    # Cosine Similarity
    cosine_sim = cosine_similarity(tfidf_matrix)

    # Create NetworkX graph
    G = create_networkx_graph(cosine_sim, threshold)
    print(G)

    # Convert to iGraph
    igraph_graph = convert_to_igraph(G)

    # Apply Leiden algorithm
    partition, leiden_time = apply_leiden_algorithm(igraph_graph)
    print("Partition", partition)
    # Plot clusters
    clusters_info = {}
    for cluster_id, nodes in enumerate(partition):
        clusters_info[cluster_id] = list(map(int, nodes))
    output_csv = r'C:\Users\Checkout\Desktop\CS255_Project\News_Clustering_Comparisons\outputs\output_time.csv'
    with open(output_csv, 'a', newline='') as csvfile:
        fieldnames = ['Algorithm', 'Execution Time', 'Number of Clusters']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Check if the file is empty and write header
        if csvfile.tell() == 0:
            writer.writeheader()

        # Write data
        writer.writerow({'Algorithm': 'Leiden', 'Execution Time': leiden_time, 'Number of Clusters': len(clusters_info)})
    print("Writing complete for Leiden")

    plot_clusters(clusters_info, G)

if __name__ == '__main__':

    '''This is the main function'''

    file_path = r'C:\Users\Checkout\Desktop\CS255_Project\News_Clustering_Comparisons\Dataset\input_file.csv'
    threshold = 0.3
    base_function(file_path, threshold)
