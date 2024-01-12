import re
from typing import Any, Iterator
from spacy.tokens import Doc, Span
from spacy.matcher import Matcher
from spacy.language import Language
from spacy.util import filter_spans
from .patterns import (
    modality_patterns,
    sf_patterns,
    connectives_patterns,
    connectives_regexes,
    connectives_classifications,
    termpp,
)


def get_connectives_classification(s: Span) -> str:
    return connectives_classifications.get(s.label_, "新")


if not Span.has_extension("connective"):
    Span.set_extension("connective", getter=get_connectives_classification)

if not Span.has_extension("modality"):
    Span.set_extension("modality", default=None)


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
    # TODO smarter filter
    spans = filter_spans(spans)
    return spans


def connectives_match(
    doc: Doc, nlp: Language, connectives_matcher: Matcher
) -> list[Span]:
    # TODO match all, but keep index of match for later filtering?
    matches = connectives_matcher(doc)
    spans = [
        Span(doc, start, end, nlp.vocab.strings[match_id])
        for match_id, start, end in matches
        if start < 8  # FIXME only look at first 8 tokens
    ]
    for sentence in doc.sents:
        # We need to index back into the doc after matching with the re module.
        char_offset = sentence.start_char

        for connective in connectives_regexes:
            # Exceptions:
            # TODO move to patterns definition
            # NOTE this will not annotate correctly
            if connective["regex"] == "が":
                rx = re.compile(r"^(" + connective["regex"] + r")[、,，]")
            else:
                rx = re.compile(connective["regex"])
            # The first matching group (m.group(1)) contains the actual tokens we want
            # to label.
            # rx = re.compile(r"^[\s　\d０-９]*「?(" + connective["regex"] + r")[、,，]")

            if m := re.search(rx, sentence.text):
                if char_offset + m.start() > 12:  # FIXME only look at first 12 chars
                    continue
                match_span = doc.char_span(
                    char_offset + m.start(),
                    char_offset + m.end(),
                    label=termpp(connective["conjunction"]),
                    alignment_mode="contract",  # "expand",
                )
                if match_span:  # If contracted can be None
                    spans.append(match_span)
    spans = filter_spans(spans)
    return spans
