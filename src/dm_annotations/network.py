import networkx as nx


def dms_to_network(xs: list[dict]) -> nx.Graph:
    G = nx.DiGraph()
    G.add_nodes_from(enumerate(xs))
    G.add_edges_from(zip(range(len(xs)), range(1, len(xs))))

    return G
