import pytest
from dm_annotations.patterns import (
    modality_patterns,
    modality_patterns_2,
    connectives_patterns,
    connectives_regexes,
    parallel_expand,
)
import spacy
from spacy.matcher import Matcher
from pyrsistent import v, m, s, pvector, thaw
from itertools import permutations


@pytest.fixture
def nlp(model="ja_ginza"):
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
    # TODO
    assert len(connectives_regexes) == 489


def pattern_is_equal(a, b):
    return any(a == pvector(variant) for variant in permutations(b))


def test_simple_patterns():
    a_token = m(a=1)
    b_token = m(b=0)
    a_vector = v(a_token)
    b_vector = v(b_token)

    # Test cases
    assert parallel_expand(a_vector) == v(a_vector)
    assert parallel_expand(v(a_vector, b_vector)) == v(v(a_token, b_token))

    # Test for OR and AND expansion combinations
    assert pattern_is_equal(
        parallel_expand(v(a_token, s(b_vector, a_vector))),
        v(v(a_token, b_token), v(a_token, a_token)),
    )
    assert pattern_is_equal(
        parallel_expand(v(a_token, s(a_vector, v(b_token, a_token)))),
        v(v(a_token, b_token, a_token), v(a_token, a_token)),
    )
    assert pattern_is_equal(
        parallel_expand(
            v(a_token, s(a_token), s(b_token, v(a_token, b_token)), v(b_token, b_token))
        ),
        v(
            v(a_token, a_token, b_token, b_token, b_token),
            v(a_token, a_token, a_token, b_token, b_token, b_token),
        ),
    )

    assert pattern_is_equal(
        parallel_expand(
            v(
                a_token,
                s(a_token, s(b_token, a_token)),
                s(b_token, v(a_token, b_token)),
                v(b_token, b_token),
            )
        ),
        v(
            v(a_token, a_token, b_token, b_token, b_token),
            v(a_token, b_token, b_token, b_token, b_token),
            v(a_token, a_token, a_token, b_token, b_token, b_token),
            v(a_token, b_token, b_token, b_token),
            v(a_token, b_token, a_token, b_token, b_token, b_token),
            v(a_token, a_token, b_token, b_token, b_token),  # Expands to same as above
            v(a_token, a_token, a_token, b_token, b_token, b_token),
        ),
    )


def test_modality_patterns_2(nlp):
    matcher = Matcher(nlp.vocab, validate=True)

    for pattern_name, d in modality_patterns_2.items():
        print(pattern_name)
        expanded_pattern = parallel_expand(d["pattern"])
        matcher.add(pattern_name, thaw(expanded_pattern))
        for example in d["examples"]:
            print(example)
            doc = nlp(example)
            matches = matcher(doc)
            assert doc and matches
