# News Clustering Algorithm

## Overview

This project implements a news clustering algorithm for grouping news articles based on their textual content. The algorithm utilizes the Louvain Modularity optimization and Leiden algorithms for community detection in graphs, along with TF-IDF vectorization and cosine similarity for text analysis.

## Prerequisites

- Python 3
- NetworkX
- igraph
- python-louvain
- NumPy
- pandas
- scikit-learn
- matplotlib


1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/your-username/news-clustering.git
    cd news-clustering
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Load your news dataset into the `Dataset` directory.
2. Update the file path in the main function of `Louvain_Modularity.py` or `Leiden_Modularity.py` based on your dataset location.
3. Run the clustering algorithm:

    ```bash
    python Louvain_Modularity.py
    ```

    or

    ```bash
    python Leiden_Modularity.py
    ```

## Contributing

Contributions are welcome! Fork the repository, create a branch, make your changes, and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


```bash
pip install -r requirements.txt
