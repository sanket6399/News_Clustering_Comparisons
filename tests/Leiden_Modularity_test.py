import os
import pytest
from News_Clustering_Comparisons.src.Leiden_Modularity import base_function

@pytest.fixture
def input_file_path():
    return r'C:\Users\Checkout\Desktop\CS255_Project\News_Clustering_Comparisons\Dataset\input_file.csv'  
    # Update with your actual input file path

def test_base_function(input_file_path):
    threshold = 0.3
    output = base_function(input_file_path, threshold)
    assert output is not None  # Checking if the base function is working

if __name__ == '__main__':
    pytest.main([os.path.basename(__file__), '-v'])
