import re
from typing import Any

from spacy.language import Language
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span
from spacy.util import filter_spans

from dm_annotations.pipeline.patterns import (
    connectives_patterns,
    connectives_regexes,
    modality_patterns,
    sf_patterns,
    termpp,
)


def create_matcher(
    patterns_dict: dict[str, dict[str, Any]], nlp: Language
) -> tuple[Language, Matcher]:
    matcher = Matcher(nlp.vocab, validate=True)

    for pattern_name, pattern_dict in patterns_dict.items():
        matcher.add(pattern_name, pattern_dict["pattern"])

    assert len(matcher) > 0

    return nlp, matcher


def create_connectives_matcher(nlp: Language) -> tuple[Language, Matcher]:
    connectives_matcher = Matcher(nlp.vocab, validate=True)

    for pattern in connectives_patterns:
        connectives_matcher.add(termpp(pattern["conjunction"]), pattern["pattern"])

    return nlp, connectives_matcher


def create_modality_matcher(nlp: Language) -> tuple[Language, Matcher]:
    """Legacy"""
    modality_matcher = Matcher(nlp.vocab, validate=True)

    for pattern_name, pattern in modality_patterns.items():
        modality_matcher.add(pattern_name, pattern)

    return nlp, modality_matcher
    # return create_matcher(modality_patterns, nlp)


def create_sf_matcher(nlp: Language) -> tuple[Language, Matcher]:
    # TODO 文の最後から名詞までヒットしたパターンでフィルター
    # - 「こと」,「もの」の後を全部抽出してみる？
    #   - 最後の7形態素
    # What to do about gaps? (Ignore??)
    return create_matcher(sf_patterns, nlp)


def filter_overlaps(matches: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    if len(matches) <= 1:
        return matches
    # Sort from start of sentence:
    sorted_matches = sorted(matches, key=lambda x: (x[1], -(x[2] - x[1])))

    filtered_matches = []
    last_end = -1
    for match in sorted_matches:
        _, start, end = match
        # If the current match starts after the last match ends, it's not overlapping:
        if start >= last_end:
            filtered_matches.append(match)
            last_end = end

    return filtered_matches


def pattern_match(doc: Doc, nlp: Language, matcher: Matcher) -> list[Span]:
    """Returns all matches for input sentence."""
    matches = matcher(doc)
    spans = [
        Span(doc, start, end, nlp.vocab.strings[match_id])
        for match_id, start, end in matches
    ]
    # Always filter overlaps so that shorter matches (like "のか") are dropped when inside longer matches
    spans = filter_spans(spans)
    return spans


def connectives_match(
    doc: Doc,
    nlp: Language,
    connectives_matcher: Matcher,
    strict: bool = True,
) -> list[Span]:
    # The doc is assumed to be a single sentence/unit.
    spans: list[Span] = []

    # Determine the start index of the first non-punct/space token.
    first_idx = 0
    try:
        sent = next(doc.sents)
        for tok in sent:
            if not (tok.is_punct or tok.is_space):
                first_idx = tok.i
                break
        else:
            first_idx = sent.start
    except StopIteration:
        # No sentences found, use doc start.
        pass

    # 1. Matcher-based patterns
    matches = connectives_matcher(doc)
    for match_id, start, end in matches:
        # Only accept matches that begin at the first real token.
        if start == first_idx:
            if not strict or (end < len(doc) and doc[end].text in {",", "、", "，"}):
                spans.append(Span(doc, start, end, nlp.vocab.strings[match_id]))

    # 2. Regex-based patterns (anchored to the start of the doc text)
    for connective in connectives_regexes:
        rx = re.compile(r"^[\s　\d０-９]*「?(" + connective["regex"] + r")[、,，]")
        if m := re.search(rx, doc.text):
            match_span = doc.char_span(
                m.start(),
                m.end(),
                label=termpp(connective["conjunction"]),
                alignment_mode="contract",
            )
            if match_span:
                spans.append(match_span)

    if not spans:
        return []

    # Find the single best connective span: earliest start, then longest.
    best_span = sorted(
        spans, key=lambda s: (s.start_char, -(s.end_char - s.start_char))
    )[0]

    return [best_span]
