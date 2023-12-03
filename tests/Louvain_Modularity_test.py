# test_louvain_modularity.py
import pytest
from src.Louvain_Modularity import louvainModularity, louvainModularityLib

@pytest.fixture
def sample_data():
    # You might need to create a sample dataset for testing
    # Ensure this data aligns with the expected format in your code
    # Example: return {'Index': [1, 2, 3], 'Headline': ['...', '...', '...'], ...}
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

