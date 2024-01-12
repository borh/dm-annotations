import pytest
import networkx as nx
import matplotlib.pyplot as plt
from dm_annotations.network import dms_to_network, visualize
from dm_annotations.corpus import extract_dm


@pytest.fixture
def G(doc, nlp):
    dms = extract_dm([doc], nlp)
    g = dms_to_network(dms)
    return g


def test_network(G):
    print(G)
    assert G
    assert G.order() == 4
    # assert G.number_of_edges(0, 1) == 1


def test_visualization(G):
    visualize(G, filename="test_dm_graph.svg")
