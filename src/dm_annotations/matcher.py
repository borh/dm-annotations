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


def create_connectives_matcher(
    model="ja_core_news_sm", nlp=None
) -> tuple[Language, Matcher]:
    if not nlp:
        nlp = spacy.load(model)

    connectives_matcher = Matcher(nlp.vocab)

    for pattern in connectives_patterns:
        connectives_matcher.add(pattern["conjunction"], pattern["pattern"])

    return nlp, connectives_matcher


def create_modality_matcher(
    model="ja_core_news_sm", nlp=None
) -> tuple[Language, Matcher]:
    if not nlp:
        nlp = spacy.load(model)

    modality_matcher = Matcher(nlp.vocab)

    for pattern_name, patterns in modality_patterns.items():
        modality_matcher.add(pattern_name, patterns)

    return nlp, modality_matcher


def modality_match(doc: Doc, nlp, modality_matcher):
    # nlp, modality_matcher = create_modality_matcher()

    matches = modality_matcher(doc)
    spans = [
        Span(doc, start, end, nlp.vocab.strings[match_id])
        for match_id, start, end in matches
        if start >= len(doc) - 10  # FIXME only look at 10 last tokens
    ]
    spans = filter_spans(spans)
    return spans


def connectives_match(doc: Doc, nlp, connectives_matcher):
    # nlp, connectives_matcher = create_connectives_matcher()
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


if __name__ == "__main__":
    nlp = spacy.load("ja_core_news_sm")
    doc = nlp(
        """後は，実験をしなければいけないのだろう。
    が，そのようなことがあるが。
    このことから，ないほうがよいでしょう。
    その結果，施策としてコミュニティ･バスによりフォーカスした群で公共交通に対する態度･行動変容効果が示唆された一方，相対的に自動車利用抑制にフォーカスした群においては，自動車利用抑制に対する態度･行動変容効果が見られ，本研究の仮説が支持されたことが示唆された．
  一般にロケットの姿勢制御においては，ロケットを剛体と近似したときの全体的な姿勢を誘導指令値に追従させるとともに，ロケットの姿勢制御系が構造振動を過度に励起しないように注意する必要がある．
"""
    )
    for t in doc:
        print(t.norm_, t.lemma_, t.pos_, t.tag_, t.morph)
    print(doc)
    print(modality_match(doc, nlp, create_modality_matcher(nlp=nlp)[1]))
    print(connectives_match(doc, nlp, create_connectives_matcher(nlp=nlp)[1]))
