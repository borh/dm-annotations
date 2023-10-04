import pytest
from dm_annotations.patterns import (
    modality_patterns,
    connectives_patterns,
    connectives_regexes,
)
import spacy
from spacy.matcher import Matcher


@pytest.fixture
def nlp(model="ja_core_news_sm"):
    nlp = spacy.load(model)
    return nlp


def test_modality_patterns(nlp):
    assert len(modality_patterns.keys()) == 63
    matcher = Matcher(nlp.vocab, validate=True)

    for pattern_name, patterns in modality_patterns.items():
        matcher.add(pattern_name, patterns)


def test_connectives_patterns(nlp):
    assert len(connectives_patterns) == 34
    matcher = Matcher(nlp.vocab, validate=True)
    for pattern in connectives_patterns:
        matcher.add(pattern["conjunction"], pattern["pattern"])


def test_connectives_regexes(nlp):
    assert len(connectives_regexes) == 489
