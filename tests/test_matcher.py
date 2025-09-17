from dm_annotations.pipeline.matcher import (
    connectives_match,
    create_connectives_matcher,
    create_matcher,
    create_modality_matcher,
    pattern_match,
)
from dm_annotations.pipeline.patterns import sf_patterns


def test_connectives_match(doc, nlp):
    nlp, matcher = create_connectives_matcher(nlp=nlp)
    assert "あと" in matcher
    matches = connectives_match(doc, nlp, matcher)
    assert matches


def test_modality_match(doc, nlp):
    nlp, matcher = create_modality_matcher(nlp=nlp)
    assert "だ" in matcher
    matches = pattern_match(doc, nlp, matcher)
    assert matches


def test_sf_match(doc, nlp):
    nlp, matcher = create_matcher(sf_patterns, nlp=nlp)
    matches = pattern_match(doc, nlp, matcher)
    assert matches


def test_debug_matcher_step_by_step(nlp):
    """Debug matcher step by step to see what's happening."""
    from dm_annotations.matcher import connectives_match, create_connectives_matcher

    text = "その結果、これは正しい。"
    doc = nlp(text)

    print(f"Text: {text}")
    print(f"Tokens: {[(t.text, t.pos_, t.tag_, t.lemma_) for t in doc]}")
    print(f"Sentences: {[sent.text for sent in doc.sents]}")

    # Test raw matcher
    _, matcher = create_connectives_matcher(nlp)
    raw_matches = matcher(doc)
    print(f"Raw matcher results: {raw_matches}")

    # Test connectives_match function
    conn_matches = connectives_match(doc, nlp, matcher)
    print(f"Connectives matches: {[(m.text, m.label_) for m in conn_matches]}")

    assert len(conn_matches) > 0, "Should find 'その結果'"
