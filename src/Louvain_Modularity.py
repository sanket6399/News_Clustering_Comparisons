from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import community.community_louvain as community_louvain
import matplotlib.pyplot as plt
import pandas as pd
import time

def load_and_preprocess_data(file_path):

    '''This functions Load and preprocess data'''

    data = pd.read_csv(file_path)
    data['text'] = data['Headline'] + " " + data['Description'] + " " + data['Article text']
    return data

def create_graph(data, threshold):

    ''' This function is used to Create graph from data'''

    G = nx.Graph()
    # Add nodes
    for idx in data['Index']:
        G.add_node(idx)
    # Add edges
    tfidf_matrix = TfidfVectorizer().fit_transform(data['text'])
    cosine_sim = cosine_similarity(tfidf_matrix)
    # the below code is to remove self similarity and similarity below threshold
    for i in range(len(cosine_sim)):
        for j in range(i + 1, len(cosine_sim)):
            if cosine_sim[i][j] > threshold:
                G.add_edge(data['Index'][i], data['Index'][j], weight=cosine_sim[i][j])

    return G

def louvain_modularity_optimized(G):

    '''This function is used to Louvain Modularity Optimized'''

    A_sparse = nx.adjacency_matrix(G)
    m = G.size(weight='weight')
    degrees = dict(G.degree(weight='weight'))
    # initial partition
    initial_partition = {node: index for index, node in enumerate(G.nodes())}
    current_partition = initial_partition.copy()
    node_to_community = initial_partition.copy()
    communities = {i: {node} for i, node in enumerate(G.nodes())}
    # while there is an improvement of modularity
    while True:
        improvement = False
        # for each node
        for node in G.nodes():
            node_community = node_to_community[node]
            # compute the gain of modularity that would result by removing the node from its community
            best_gain = 0
            best_community = node_community
            # for each neighbor of node
            for neighbor in G.neighbors(node):
                # compute the gain of modularity that would result by removing the node from its community
                community = node_to_community[neighbor]
                if community != node_community:
                    new_partition = current_partition.copy()
                    # remove node from its community
                    new_partition[node] = community
                    gain = calculate_modularity_optimized(G, new_partition, A_sparse, m, degrees) - calculate_modularity_optimized(G, current_partition, A_sparse, m, degrees)
                    if gain > best_gain:
                        best_gain = gain
                        best_community = community
            # move the node to the community that results in the maximum gain
            if best_community != node_community:
                communities[node_community].remove(node)
                communities[best_community].add(node)
                current_partition[node] = best_community
                improvement = True
                node_to_community[node] = best_community
        # if there was no improvement, terminate
        if not improvement:
            break

    # return the partition
    return {node: current_partition[node] for node in G.nodes()}

def calculate_modularity_optimized(G, partition, A, m, degrees):

    '''Calculate modularity based on partition'''

    # compute the modularity
    modularity = 0.0
    node_index = {node: idx for idx, node in enumerate(G.nodes())}

    # for each pair of nodes
    for i in G.nodes():
        for j in G.neighbors(i):
            # if the nodes belong to the same community
            if partition[i] == partition[j]:
                modularity += A[node_index[i], node_index[j]] - degrees[i] * degrees[j] / (2 * m)

    # return the modularity
    return modularity / (2 * m)

def plot_clusters(clusters, G):

    '''This function is used to Plot clusters'''

    i = 0
    for cluster_id, nodes in clusters.items():
        # print("Cluster", i, ":", nodes)
        subgraph = G.subgraph(nodes)
        pos_subgraph = nx.spring_layout(subgraph)

        plt.figure(figsize=(8, 6))
        nx.draw(subgraph, pos_subgraph, node_color='yellow', with_labels=True, font_size=8, node_size=280)

        labels_subgraph = {node: str(node) for node in subgraph.nodes}
        nx.draw_networkx_labels(subgraph, pos_subgraph, labels_subgraph, font_size=8, font_color='black')

        plt.title(f"Cluster {i}")
        i += 1
        plt.show()

def louvain_modularity_wrapper(file_path, threshold = 0.32):

    '''This function is used as a Wrapper for Louvain Modularity without library'''

    start_time = time.time()
    # Load and preprocess data
    data = load_and_preprocess_data(file_path)
    G = create_graph(data, threshold)
    # Apply Louvain algorithm
    partition_optimized = louvain_modularity_optimized(G)

    end_time = time.time()
    louvain_time = -1 * (start_time - end_time)
    print("Louvain time", louvain_time)
    print("Partition", partition_optimized)

    # create clusters
    clusters = {}
    for node, cluster_id in partition_optimized.items():
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(node)

    # print("Cluster", i, ":", nodes)
    list(clusters.items())
    print("Number of clusters formed: ", len(clusters))
    plot_clusters(clusters, G)

def louvainModularityLib(file_path):

    '''This function is Louvain Modularity with library'''

    data = pd.read_csv(file_path)
    # getting the columns data inside my df
    data['text'] = data['Headline'] + " " + data['Description'] + " " + data['Article text']
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['text'])
    cosine_sim = cosine_similarity(tfidf_matrix)

    # Create NetworkX graph
    G = nx.Graph()
    for idx in data['Index']:
        G.add_node(idx)

    threshold = 0.3

    # the below code is to remove self similarity and similarity below threshold
    for i in range(len(cosine_sim)):
        for j in range(i + 1, len(cosine_sim)):
            if cosine_sim[i][j] > threshold:
                G.add_edge(data['Index'][i], data['Index'][j], weight=cosine_sim[i][j])

    # Apply Louvain algorithm
    partition = community_louvain.best_partition(G)
    clusters = {}

    # create clusters
    for node, cluster_id in partition.items():
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(node)

    # print("Cluster", i, ":", nodes)
    list(clusters.items())
    print("Number of clusters formed: ", len(clusters))
    plot_clusters(clusters, G)

if __name__ == '__main__':
    # print("Louvain Modularity without library")

    # Change the file path to the location of the dataset on your machine

    # file_path_small = r"C:\Users\Checkout\Desktop\CS255_Project\News_Clustering_Comparisons\Dataset\sample_small.csv"
    # louvain_modularity_wrapper(file_path_small)
    print("Louvain Modularity with library")
    file_path = r'C:\Users\Checkout\Desktop\CS255_Project\News_Clustering_Comparisons\Dataset\input_file.csv'
    louvainModularityLib(file_path)
