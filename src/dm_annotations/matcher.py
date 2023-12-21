import re
import spacy
from spacy.tokens import Doc, Span
from spacy.matcher import Matcher
from spacy.util import filter_spans
from spacy.language import Language
from .patterns import (
    modality_patterns,
    connectives_patterns,
    connectives_regexes,
    connectives_classifications,
)


def get_connectives_classification(s: Span) -> str:
    return connectives_classifications.get(s.label_, "新")


if not Span.has_extension("connective"):
    Span.set_extension("connective", getter=get_connectives_classification)

if not Span.has_extension("modality"):
    Span.set_extension("modality", default=None)


def create_matcher(
    patterns_dict: dict[str, list], model="ja_ginza", nlp=None
) -> tuple[Language, Matcher]:
    if not nlp:
        nlp = spacy.load(model)

    matcher = Matcher(nlp.vocab)

    for pattern_name, pattern in patterns_dict.items():
        matcher.add(pattern_name, pattern)

    return nlp, matcher


def create_connectives_matcher(model="ja_ginza", nlp=None) -> tuple[Language, Matcher]:
    if not nlp:
        nlp = spacy.load(model)

    connectives_matcher = Matcher(nlp.vocab)

    for pattern in connectives_patterns:
        connectives_matcher.add(pattern["conjunction"], pattern["pattern"])

    return nlp, connectives_matcher


def create_modality_matcher(model="ja_ginza", nlp=None) -> tuple[Language, Matcher]:
    return create_matcher(modality_patterns, model, nlp)


def modality_match(doc: Doc, nlp, modality_matcher):
    matches = modality_matcher(doc)
    spans = [
        Span(doc, start, end, nlp.vocab.strings[match_id])
        for match_id, start, end in matches
        if start >= len(doc) - 10  # FIXME only look at 10 last tokens
    ]
    spans = filter_spans(spans)
    return spans


def connectives_match(doc: Doc, nlp, connectives_matcher):
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
                    label=connective["conjunction"],
                    alignment_mode="contract",  # "expand",
                )
                if match_span:  # If contracted can be None
                    spans.append(match_span)
    spans = filter_spans(spans)
    return spans
