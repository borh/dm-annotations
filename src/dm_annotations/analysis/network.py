from itertools import groupby
from math import log2, log10
from operator import itemgetter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import polars as pl
from community import community_louvain  # type: ignore[import]
from matplotlib import rcParams
from netgraph import Graph  # type: ignore[import]

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["IBM Plex Sans JP"]
rcParams["svg.fonttype"] = "none"


def total_edge_weight(node, graph):
    weight = sum(
        graph.get_edge_data(u, v)["weight"]
        for u, v in graph.edges(node)
        if graph.has_edge(u, v)
    )
    weight += sum(
        graph.get_edge_data(u, v)["weight"]
        for u, v in graph.in_edges(node)
        if graph.has_edge(u, v)
    )
    return weight


def calculate_pmi(graph, node1, node2, total_edges):
    # Calculate joint probability
    joint_prob = (
        graph.get_edge_data(node1, node2)["weight"] / total_edges
        if graph.has_edge(node1, node2)
        else 0.0
    )

    # Calculate individual probabilities
    prob_node1 = total_edge_weight(node1, graph) / total_edges
    prob_node2 = total_edge_weight(node2, graph) / total_edges

    # Calculate PMI
    pmi = log2(joint_prob / (prob_node1 * prob_node2)) if joint_prob > 0.0 else 0.0
    return pmi


def calculate_entropy(graph, node):
    total_weight = total_edge_weight(node, graph)
    if total_weight == 0:
        return 0.0

    entropy = 0.0
    for neighbor in graph.neighbors(node):
        edge_weight = graph.get_edge_data(node, neighbor)["weight"]
        prob = edge_weight / total_weight
        entropy -= prob * log2(prob)
    return entropy


def filter_graph_by_pmi_entropy_and_frequency(G, min_pmi, min_entropy, min_freq):
    # Remove edges with PMI below the threshold
    for u, v, data in list(G.edges(data=True)):
        if "pmi" in data and data["pmi"] < min_pmi:
            G.remove_edge(u, v)

    # Remove nodes with entropy below the threshold or frequency below the threshold
    for node, data in list(G.nodes(data=True)):
        if ("entropy" in data and data["entropy"] < min_entropy) or (
            "frequency" in data and data["frequency"] < min_freq
        ):
            G.remove_node(node)

    # Optionally, remove isolated nodes
    for node in list(nx.isolates(G)):
        G.remove_node(node)


def dms_to_network(
    dm_seq: list[dict[str, Any]],
    min_pmi: float = 0.0,
    min_entropy: float = 0.0,
    min_freq: int = 0,
) -> nx.DiGraph:
    """
    Build a directed graph from a flat sequence of DM‐match dicts.

    Nodes are expressions, edges connect consecutive matches in each sentence.
    PMI, entropy, and frequency are computed on the fly.

    :param dm_seq: list of dicts with keys "sentence_id","ジャンル","title","表現","タイプ"
    :param min_pmi: drop edges with PMI < threshold
    :param min_entropy: drop nodes with entropy < threshold
    :param min_freq: drop nodes with frequency < threshold
    :return: filtered `networkx.DiGraph`

    >>> dms = [
    ...   {"sentence_id":0,"ジャンル":"G","title":"T","表現":"A","タイプ":"t"},
    ...   {"sentence_id":0,"ジャンル":"G","title":"T","表現":"B","タイプ":"t"},
    ... ]
    >>> G = dms_to_network(dms)
    >>> G.has_edge("A","B")
    True
    """
    G: nx.DiGraph = nx.DiGraph()
    # Pull the sequence into a list so we can loop twice (nodes + sorted for edges)
    dm_records = list(dm_seq)
    # Ensure every DM expression shows up as a node, even if no edges form
    for rec in dm_records:
        expr = rec["表現"]
        typ = rec.get("タイプ", "")
        if expr not in G:
            G.add_node(expr, type=typ, entropy=0.0, frequency=1)
        else:
            G.nodes[expr]["frequency"] += 1
    sorted_dm_seq = sorted(
        dm_records, key=itemgetter("sentence_id", "ジャンル", "title")
    )
    for _, sentence_group in groupby(
        sorted_dm_seq, key=itemgetter("sentence_id", "ジャンル", "title")
    ):
        sentence_dms = list(sentence_group)
        for i in range(len(sentence_dms) - 1):
            source_span = sentence_dms[i]["表現"]
            target_span = sentence_dms[i + 1]["表現"]

            source_type = sentence_dms[i]["タイプ"]
            target_type = sentence_dms[i + 1]["タイプ"]

            if source_span not in G:
                G.add_node(source_span, type=source_type, entropy=0, frequency=1)
            else:
                G.nodes[source_span]["frequency"] += 1

            if target_span not in G:
                G.add_node(target_span, type=target_type, entropy=0, frequency=1)
            else:
                G.nodes[target_span]["frequency"] += 1

            total_edges = sum([G[u][v]["weight"] for u, v in G.edges()])
            if G.has_edge(source_span, target_span):
                G[source_span][target_span]["weight"] += 1
            else:
                G.add_edge(source_span, target_span, weight=1, pmi=0)

            # Update PMI for the edge
            pmi_value = calculate_pmi(G, source_span, target_span, total_edges + 1)
            G[source_span][target_span]["pmi"] = pmi_value

            # Update entropy for the nodes
            G.nodes[source_span]["entropy"] = calculate_entropy(G, source_span)
            G.nodes[target_span]["entropy"] = calculate_entropy(G, target_span)

    # Filter the graph based on PMI, entropy, and frequency thresholds
    filter_graph_by_pmi_entropy_and_frequency(G, min_pmi, min_entropy, min_freq)

    return G


def export_network(G, filename="dm_graph.net"):
    nx.write_pajek(G, filename)


def visualize(
    G,
    min_pmi=0.0,
    min_entropy=0.0,
    min_freq: int = 0,
    min_weight: int = -1,
    min_degree: int = -1,
    filename="dm_graph.svg",
):
    # Find nodes with fewer edges than the threshold
    nodes_with_few_edges = [node for node, degree in G.degree() if degree < min_degree]

    # Find nodes where all connected edges have weights below the threshold
    nodes_with_low_weight = [
        node for node in G.nodes() if total_edge_weight(node, G) < min_weight
    ]

    # Combine the lists of nodes to remove (avoiding duplicates)
    nodes_to_remove = list(set(nodes_with_few_edges + nodes_with_low_weight))

    filter_graph_by_pmi_entropy_and_frequency(G, min_pmi, min_entropy, min_freq)

    # Create a subgraph without these nodes
    H = G.copy()
    H.remove_nodes_from(nodes_to_remove)

    U = H.to_undirected()
    fig, ax = plt.subplots()
    node_to_community = community_louvain.best_partition(U)

    # https://matplotlib.org/stable/users/explain/colors/colors.html
    community_to_color = {
        0: "tab:blue",
        1: "tab:orange",
        2: "tab:green",
        3: "tab:red",
        4: "tab:purple",
        5: "tab:brown",
        6: "tab:pink",
        7: "tab:gray",
        8: "tab:olive",
        9: "tab:cyan",
        10: "xkcd:navy blue",
        11: "xkcd:beige",
        12: "xkcd:salmon",
        13: "xkcd:lime",
    }
    node_color = {
        node: community_to_color.get(community_id, "xkcd:deep brown")
        for node, community_id in node_to_community.items()
    }
    edge_weights = {(u, v): log10(U[u][v]["weight"]) for u, v in U.edges()}
    print(edge_weights)
    Graph(
        U,
        node_labels=True,
        node_color=node_color,
        node_edge_width=0,
        edge_alpha=0.8,
        edge_width=edge_weights,
        node_layout="community",
        node_layout_kwargs=dict(node_to_community=node_to_community),
        edge_layout="bundled",
        # edge_layout_kwargs=dict(k=2000),
        ax=ax,
    )
    fig.savefig(Path(filename).stem + "-community.pdf", format="pdf")
    plt.close(fig)
    plt.clf()

    fig, ax = plt.subplots()
    edge_weights = {(u, v): log10(H[u][v]["weight"]) for u, v in H.edges()}
    node_type_colormap = {
        "接続表現": "tab:blue",
        "文末表現": "tab:orange",
    }
    node_color = {n: node_type_colormap[H.nodes[n]["type"]] for n in H.nodes()}
    Graph(
        H,
        node_layout="radial",
        edge_width=edge_weights,
        edge_color="black",
        node_labels=True,
        node_edge_width=0,
        node_color=node_color,
        arrows=True,
        ax=ax,
    )
    fig.savefig(Path(filename).stem + "-directed.pdf", format="pdf")

    # agraph = nx.drawing.nx_agraph.to_agraph(G)
    # colors = {"sf": "red", "c": "magenta"}
    # for node in G.nodes:
    #     # agraph.get_node(node).attr["label"] = G.nodes[node]["name"]
    #     agraph.get_node(node).attr["color"] = colors[G.nodes[node]["type"]]
    # # Set edge attributes (including labels for weights)
    # for edge in G.edges:
    #     weight = G.edges[edge]["weight"]
    #     if weight >= weight_threshold:
    #         agraph_edge = agraph.get_edge(edge[0], edge[1])
    #         agraph_edge.attr["label"] = str(weight)
    #         # Optional: Customize edge label style, e.g., font size/color
    #         # agraph_edge.attr['fontsize'] = 10
    #         # agraph_edge.attr['fontcolor'] = 'blue'
    # agraph.draw(filename, format="svg", prog="dot")


if __name__ == "__main__":
    import polars as pl

    df = pl.read_csv("learner-dms.csv")
    G = dms_to_network(df.to_dicts())
    export_network(G, filename="learner-dms.net")

    visualize(
        G,
        min_pmi=2.0,
        min_entropy=0.1,
        min_freq=20,
        min_weight=100,
        min_degree=5,
        filename="learner-dms",
    )

    df = pl.read_csv("science-dms.csv")
    G = dms_to_network(df.to_dicts())
    export_network(G, filename="science-dms.net")

    visualize(
        G,
        min_pmi=2.0,
        min_entropy=0.1,
        min_freq=500,
        filename="science-dms",
    )
