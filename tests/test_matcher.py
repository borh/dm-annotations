from dm_annotations.matcher import (
    connectives_match,
    pattern_match,
    create_connectives_matcher,
    create_modality_matcher,
    create_matcher,
)
from dm_annotations.patterns import sf_patterns


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
