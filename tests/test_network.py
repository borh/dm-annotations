import pytest
import networkx as nx
import matplotlib.pyplot as plt
from dm_annotations.network import dms_to_network


@pytest.fixture
def G():
    d = [{"type": "connective", "name": "しかし"}, {"type": "modality", "name": "かもしれない"}]
    g = dms_to_network(d)
    return g


def test_network(G):
    assert G
    assert G.nodes[0]["type"] == "connective"
    assert G.order() == 2
    assert G.number_of_edges(0, 1) == 1


def test_visualization(G):
    agraph = nx.drawing.nx_agraph.to_agraph(G)
    colors = {"modality": "red", "connective": "magenta"}
    for node in G.nodes:
        agraph.get_node(node).attr["label"] = G.nodes[node]["name"]
        agraph.get_node(node).attr["color"] = colors[G.nodes[node]["type"]]
    agraph.draw("connective_modality_graph.svg", format="svg", prog="dot")
