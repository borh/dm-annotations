from itertools import permutations

from ginza import force_using_normalized_form_as_lemma
from pyrsistent import m, pvector, s, thaw, v
import pytest
import spacy
from spacy.matcher import Matcher

from dm_annotations.patterns import (
    connectives_patterns,
    connectives_regexes,
    modality_patterns,
    modality_patterns_2,
    parallel_expand,
    sf_definitions,
    sf_patterns,
    split_defs,
)
from dm_annotations.matcher import filter_overlaps

# force_using_normalized_form_as_lemma(True)


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


def test_connectives_regexes():
    assert len(connectives_regexes) == 489


def test_classifications():
    assert split_defs("このころ（この頃）、このごろ（この頃）") == {
        "このころ",
        "このごろ",
        "この頃",
        "このころ（この頃）",
        "このごろ（この頃）",
        "この頃、このごろ",
    }
    assert split_defs("そのあと、そのご（その後）") == {
        "そのあと",
        "そのご",
        "そのご（その後）",
        "そのあと、そのご",
        "その後",
    }


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

    # assert pattern_is_equal(
    #     parallel_expand(
    #         v(
    #             a_token,
    #             s(a_token, s(b_token, a_token)),
    #             s(b_token, v(a_token, b_token)),
    #             v(b_token, b_token),
    #         )
    #     ),
    #     v(
    #         v(a_token, a_token, b_token, b_token, b_token),
    #         v(a_token, b_token, b_token, b_token, b_token),
    #         v(a_token, a_token, a_token, b_token, b_token, b_token),
    #         v(a_token, b_token, b_token, b_token),
    #         v(a_token, b_token, a_token, b_token, b_token, b_token),
    #         v(a_token, a_token, b_token, b_token, b_token),  # Expands to same as above
    #         v(a_token, a_token, a_token, b_token, b_token, b_token),
    #     ),
    # )


def test_modality_patterns_2(nlp):
    matcher = Matcher(nlp.vocab, validate=True)
    for pattern_name, d in modality_patterns_2.items():
        single_matcher = Matcher(nlp.vocab, validate=True)
        print(pattern_name)
        expanded_pattern = parallel_expand(d["pattern"])
        single_matcher.add(pattern_name, thaw(expanded_pattern))
        matcher.add(pattern_name, thaw(expanded_pattern))
        for example in d["examples"]:
            print(example)
            doc = nlp(example)
            matches = single_matcher(doc)
            assert [(t.norm_, t.lemma_, t.pos_) for t in doc] and matches

    for pattern_name, d in modality_patterns_2.items():
        for example in d["examples"]:
            doc = nlp(example)
            matches = matcher(doc)
            assert doc and matches
            assert doc and any(
                nlp.vocab.strings[match_id] == pattern_name
                for match_id, _, _ in matches
            )


def test_sf_patterns(nlp):
    # 大分類
    assert (
        len(set(p["category"][0] for _, p in sf_definitions.items())) == 11
    )  # NOTE: 10 + NA
    # 再分類
    assert len(set(p["category"][1] for _, p in sf_definitions.items())) == 40
    # パターン数
    assert len(sf_definitions) == 105

    matcher = Matcher(nlp.vocab, validate=True)
    for pattern_name, d in sf_patterns.items():
        single_matcher = Matcher(nlp.vocab, validate=True)
        single_matcher.add(pattern_name, d["pattern"])
        matcher.add(pattern_name, d["pattern"])
        for example in d["examples"]:
            doc = nlp(example)
            matches = single_matcher(doc)
            if not matches:
                print(doc, matches)
            assert [
                (t.norm_, t.lemma_, t.pos_, t.tag_, t.morph) for t in doc
            ] and matches
        for example in d.get("negative_examples", []):
            doc = nlp(example)
            matches = single_matcher(doc)
            if matches:
                print(doc, matches)
            assert [(t.norm_, t.lemma_, t.pos_) for t in doc] and not matches

    # Use full matcher
    for pattern_name, d in sf_patterns.items():
        for example in d["examples"]:
            doc = nlp(example)
            matches = matcher(doc)
            print(
                [nlp.vocab.strings[match_id] for match_id, _, _ in matches],
            )
            matches = filter_overlaps(matches)
            print(
                example,
                pattern_name,
                [nlp.vocab.strings[match_id] for match_id, _, _ in matches],
            )
            assert doc and matches
            assert doc and any(
                nlp.vocab.strings[match_id] == pattern_name
                for match_id, _, _ in matches
            )
            assert doc and len(matches) == 1  # This is the only match.
