import pandas as pd
import nltk
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer
import re
import matplotlib.pyplot as plt
import csv
import time
import networkx as nx
from sklearn.cluster import SpectralClustering
nltk.download('punkt')
nltk.download('stopwords')

def load_data(csv_file):
    """This methods loads data from a csv file"""
    data = pd.read_csv(csv_file)
    return data

def preprocess(text):
    """Precprocess the text to make it for vectorization"""
    stemmer = PorterStemmer()
    stopwords = set(nltk.corpus.stopwords.words('english'))
    text = text.lower()
    # Remove whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Remove stopwords
    tokens = [token for token in tokens if token not in stopwords]
    # Stemming
    tokens = [stemmer.stem(token) for token in tokens]
    # Join cleaned tokens
    clean_text = ' '.join(tokens)
    return clean_text

def compute_text_similarity(text1, text2):
    """computes simialrity score of the two text"""
    # Preprocess text
    text1 = preprocess(text1)
    text2 = preprocess(text2)

    # Extract TF-IDF vectors
    vectorizer = TfidfVectorizer()
    vec1 = vectorizer.fit_transform([text1])
    vec2 = vectorizer.transform([text2])

    # Compute cosine similarity only for complete article text
    text_sim = cosine_similarity(vec1, vec2)[0][0]
    return text_sim

class Node:
    def __init__(self, index, author, date_published, category, section, url, headline, description, keywords, second_headline,
                 text):
        self.Index = index
        self.author = author
        self.date_published = date_published
        self.category = category
        self.section = section
        self.url = url
        self.headline = headline
        self.description = description
        self.keywords = keywords
        self.second_headline = second_headline
        self.text = text

def connect_nodes(Nodes):
    """creates a graph using the node list"""
  G = nx.Graph()
  for node1 in Nodes:
    for node2 in Nodes:
      if node1 != node2 and (G.has_edge(node1,node2)==False) and (G.has_edge(node2,node1)==False):
        if node1.category == node2.category:
          similarity = max(compute_text_similarity(node1.text, node2.text),compute_text_similarity(node2.text, node1.text))
        else:
          similarity = min(compute_text_similarity(node1.text, node2.text),compute_text_similarity(node2.text, node1.text))
        if similarity >= 0.42:
          G.add_edge(node1, node2, attr_dict = {'distance':similarity})
  return G



def build_initial_graph():
    """loads data from a input csv file and builds graph to be used as input for dp clustering algorithm"""
    data = load_data("/ip.csv")
    Node_list = []
    for i in range(0, len(data)):
        temp = Node(data.iloc[i][0], data.iloc[i][1], data.iloc[i][2], data.iloc[i][3], data.iloc[i][4],
        data.iloc[i][5], data.iloc[i][6], data.iloc[i][7], data.iloc[i][8], data.iloc[i][9], data.iloc[i][10])
        Node_list.append(temp)
    graph = connect_nodes(Node_list)
    return graph

def compute_node_similarity(text1, text2, category1, category2):
    """computes similarity score of two nodes"""
    # Preprocess text
    text1 = preprocess(text1)
    text2 = preprocess(text2)

    # Extract TF-IDF vectors
    vectorizer = TfidfVectorizer()
    vec1 = vectorizer.fit_transform([text1])
    vec2 = vectorizer.transform([text2])

    # Compute cosine similarity only for complete article text
    text_sim1 = cosine_similarity(vec1, vec2)[0][0]
    text_sim2 = cosine_similarity(vec2, vec1)[0][0]

    similarity = max(text_sim1, text_sim2)
    if(category1==category2):
      similarity = similarity+0.3

    return similarity

def has_node_changed(node, previous_degrees, current_graph):
    """this method checks if state of graph(degree) has changed"""
    current_degree = current_graph.degree(node)
    previous_degree = previous_degrees.get(node, None)
    return previous_degree is not None and current_degree != previous_degree

def update_supernodes(G, previous_degrees, existing_supernodes):
    """updates supernodes based on the new node added"""
    new_supernodes = {}
    for supernode_id, nodes in existing_supernodes.items():
        unchanged_nodes = [node for node in nodes if not has_node_changed(node, previous_degrees, G)]
        if unchanged_nodes:
            new_supernodes[supernode_id] = unchanged_nodes
    return new_supernodes

def reduce_graph(G, theta):
    """reduces teh nodes that have degree less than theta"""
    reduced_graph = nx.Graph(G)
    for node in list(reduced_graph.nodes()):
        if reduced_graph.degree(node) < theta:
            reduced_graph.remove_node(node)
    return reduced_graph

def apply_spectral_clustering(G, num_clusters):
    """applies spectral clustering on the graph and generates clusters"""
    adjacency_matrix = nx.to_numpy_array(G)

    sc = SpectralClustering(n_clusters=num_clusters, affinity='precomputed', n_init=100, assign_labels='discretize')
    labels = sc.fit_predict(adjacency_matrix)

    clusters = {}
    for index, label in enumerate(labels):
        clusters.setdefault(label, []).append(list(G.nodes())[index])

    return clusters

def dpocg_algorithm(G, previous_degrees, existing_supernodes, theta=2, num_clusters=17):
    """this method runs the dp algortihm for graph clustering """
    updated_supernodes = update_supernodes(G, previous_degrees, existing_supernodes)
    reduced_graph = reduce_graph(G, theta)
    clusters = apply_spectral_clustering(reduced_graph, num_clusters)
    updated_degrees = {node: G.degree(node) for node in G.nodes()}
    return clusters, updated_degrees, updated_supernodes

initial_graph = build_initial_graph()

cls=None

def run_dp_algorithm():
    """"""
    csv_file = 'PATH TO INPUT CSV FILE'
    df = pd.read_csv(csv_file)

    # Store data of existing nodes (if necessary)
    previous_data = {}
    cls=None
    previous_degrees = {node: initial_graph.degree(node) for node in initial_graph.nodes()}
    supernodes = {}
    count=1
    # Iterate through CSV rows, adding each as a new node
    for index, row in df.iterrows():
        new_node_text = row['Article text']
        new_node_category = row['Category']
        new_node = Node(row['Index'],row['Author'],row['Date published'],row['Category'],row['Section'],row['Url'],row['Headline'],row['Description'],row['Keywords'],row['Second headline'],row['Article text'])

        initial_graph.add_node(new_node)

        for existing_node in initial_graph.nodes():
            if existing_node != new_node:
                # Assume 'text' attribute stores the content of each node
                existing_node_category = existing_node.category
                existing_node_text = existing_node.text
                similarity = compute_node_similarity(new_node_text, existing_node_text, new_node_category, existing_node_category)
                if similarity > 0.42:  # Example threshold for adding an edge
                    initial_graph.add_edge(new_node, existing_node, weight=similarity)

        previous_degrees[new_node] = initial_graph.degree(new_node)
        clusters, previous_degrees, supernodes = dpocg_algorithm(initial_graph, previous_degrees, supernodes)
        cls = clusters
        count=count+1
    return cls

start_time = time.time()
clusters = run_dp_algorithm()
end_time = time.time()
dp_time = end_time-start_time
output_csv = '/News_Clustering_Comparisons/output_time.csv'
with open(output_csv, 'a', newline='') as csvfile:
    fieldnames = ['Algorithm', 'Execution Time', 'Number of Clusters']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    # Check if the file is empty and write header
    if csvfile.tell() == 0:
        writer.writeheader()
    # Write data
    writer.writerow({'Algorithm': 'Dynamic Programming', 'Execution Time': dp_time, 'Number of Clusters': len(clusters)})


