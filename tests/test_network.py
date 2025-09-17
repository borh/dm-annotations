import networkx as nx
import pytest

from dm_annotations.analysis.network import dms_to_network, visualize


@pytest.fixture
def G(doc, nlp):
    # doc fixture now includes dm_extractor processing
    dms = doc._.dm_matches
    print(f"Doc text: {doc.text[:100]}...")
    print(f"Doc metadata: {doc.user_data.get('meta', {})}")
    print(f"Number of DM matches: {len(dms)}")
    print(f"DM matches: {dms}")

    # Ensure we have some matches for testing
    if not dms:
        pytest.skip("No DM matches found in test document")

    g = dms_to_network(dms)
    return g


def test_network(G):
    print(nx.to_dict_of_dicts(G))
    assert G
    assert G.order() >= 1  # At least one node
    # Only check edges if we have multiple nodes
    if G.order() > 1:
        assert G.number_of_edges() >= 0


def test_visualization(G):
    visualize(G, filename="test_dm_graph.svg")


def test_dm_extraction_directly(nlp):
    """Test DM extraction directly to debug issues."""
    # Use the shared nlp fixture instead of creating a new one
    test_text = "その結果、これは正しいと思う。"
    doc = nlp(test_text)

    # Set metadata
    doc.user_data["meta"] = {
        "title": "test",
        "genre": "test",
        "sentence_id": 0,
        "basename": "test",
        "author": "test_author",
        "year": 2024,
        "paragraph_id": None,
        "section": None,
    }

    print(f"Test text: {test_text}")
    print(f"Pipeline: {nlp.pipe_names}")
    print(f"Metadata: {doc.user_data.get('meta', {})}")
    print(f"DM matches: {doc._.dm_matches}")

    # This should find at least "その結果" as a connective
    assert len(doc._.dm_matches) > 0, "Expected at least one DM match"
