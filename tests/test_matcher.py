import pytest
import spacy
from dm_annotations.matcher import (
    connectives_match,
    modality_match,
    create_connectives_matcher,
    create_modality_matcher,
    create_matcher,
)
from dm_annotations.patterns import modality_patterns_2, parallel_expand
from pyrsistent import thaw


@pytest.fixture(scope="module")
def nlp():
    return spacy.load("ja_ginza")


@pytest.fixture(scope="module")
def doc(nlp):
    return nlp(
        """
その結果，施策としてコミュニティ･バスによりフォーカスした群で公共交通に対する態度･行動変容効果が示唆された一方，相対的に自動車利用抑制にフォーカスした群においては，自動車利用抑制に対する態度･行動変容効果が見られ，本研究の仮説が支持されたことが示唆された．"""
    )


def test_connectives_match(doc, nlp):
    nlp, matcher = create_connectives_matcher(nlp=nlp)
    assert "あと" in matcher
    matches = connectives_match(doc, nlp, matcher)
    assert matches


def test_modality_match(doc, nlp):
    nlp, matcher = create_modality_matcher(nlp=nlp)
    assert "だ" in matcher
    matches = modality_match(doc, nlp, matcher)
    assert matches


def test_modality_2_match(doc, nlp):
    patterns = {
        pattern_name: thaw(parallel_expand(d["pattern"]))
        for pattern_name, d in modality_patterns_2.items()
    }
    nlp, matcher = create_matcher(patterns, nlp=nlp)
    matches = modality_match(doc, nlp, matcher)
    assert matches
