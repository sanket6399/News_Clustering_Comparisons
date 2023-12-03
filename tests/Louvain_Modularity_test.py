# test_louvain_modularity.py
import pytest
from News_Clustering_Comparisons.src.Leiden_Modularity import louvainModularity, louvainModularityLib

@pytest.fixture
def sample_data():
    pass

def test_louvain_modularity(sample_data, capsys):
    # Capture the printed output
    with capsys.disabled():
        louvainModularity()
    # Assert something based on the output (e.g., check for expected console prints)

def test_louvain_modularity_lib(sample_data, capsys):
    # Capture the printed output
    with capsys.disabled():
        louvainModularityLib()
    # Assert something based on the output (e.g., check for expected console prints)

