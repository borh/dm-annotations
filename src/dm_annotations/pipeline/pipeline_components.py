from ginza import bunsetu_spans
from spacy.language import Language
from spacy.tokens import Doc, Span

from dm_annotations.io.loader import normalize_meta
from dm_annotations.pipeline.matcher import (
    connectives_match,
    create_connectives_matcher,
    create_sf_matcher,
    pattern_match,
)
from dm_annotations.pipeline.patterns import (
    connectives_classifications,
    sf_classifications,
)
from dm_annotations.schemas import DmMatch, DocMetadata


def filter_sf_final_bunsetu(spans: list[Span], doc: Doc) -> list[Span]:
    """
    Keep only SF spans in or connected to the final bunsetsu of the Doc.
    """
    bs = list(bunsetu_spans(doc))
    if not bs:
        return spans

    target = {len(bs) - 1}
    filtered: list[Span] = []
    added = True
    while added:
        added = False
        for span in spans:
            idxs = {
                i for i, b in enumerate(bs) if span.start < b.end and span.end > b.start
            }
            if idxs & target and span not in filtered:
                filtered.append(span)
                target |= idxs
                added = True

    return sorted(filtered, key=lambda s: s.start)


class DmConnectivesComponent:
    def __init__(self, nlp: Language, name: str):
        self.nlp, self.matcher = create_connectives_matcher(nlp)

    def __call__(self, doc: Doc) -> Doc:
        meta = DocMetadata(**normalize_meta(doc.user_data.get("meta")))
        matches: list[DmMatch] = [
            DmMatch(
                span=span,
                表現=span.label_,
                タイプ="接続表現",
                機能=connectives_classifications[span.label_][0],
                細分類=connectives_classifications[span.label_],
                position=span.start_char / len(doc.text),
                ジャンル=meta.genre,
                title=meta.title,
                sentence_id=meta.sentence_id,
            )
            for span in connectives_match(doc, self.nlp, self.matcher)
        ]
        doc._.dm_connectives = matches
        return doc


class DmSfComponent:
    def __init__(self, nlp: Language, name: str):
        self.nlp, self.matcher = create_sf_matcher(nlp)

    def __call__(self, doc: Doc) -> Doc:
        meta = DocMetadata(**normalize_meta(doc.user_data.get("meta")))
        matches: list[DmMatch] = [
            DmMatch(
                span=span,
                表現=span.label_,
                タイプ="文末表現",
                機能=sf_classifications[span.label_][0],
                細分類=sf_classifications[span.label_][1],
                position=span.start_char / len(doc.text),
                ジャンル=meta.genre,
                title=meta.title,
                sentence_id=meta.sentence_id,
            )
            for span in pattern_match(doc, self.nlp, self.matcher)
        ]
        doc._.dm_sf = matches
        return doc


class DmExtractor:
    """One spaCy pipe extracting both connectives and sentence-final patterns."""

    def __init__(
        self,
        nlp,
        match_kinds: tuple[str, str] = ("connectives", "sf"),
        sf_final_filter: bool = True,
        strict_connectives: bool = False,
    ):
        self.nlp = nlp
        self.matchers = {}
        self.sf_final_filter = sf_final_filter
        self.strict_connectives = strict_connectives
        if "connectives" in match_kinds:
            _, conn = create_connectives_matcher(nlp)
            self.matchers["connectives"] = conn
        if "sf" in match_kinds:
            _, sfm = create_sf_matcher(nlp)
            self.matchers["sf"] = sfm

    def __call__(self, doc: Doc) -> Doc:
        out: list[DmMatch] = []

        # Normalize metadata via DocMetadata
        meta = DocMetadata(**normalize_meta(doc.user_data.get("meta")))
        sid = meta.sentence_id
        genre = meta.genre
        title = meta.title
        section = meta.section

        connective_spans: list[Span] = []
        if "connectives" in self.matchers:
            connective_spans = connectives_match(
                doc, self.nlp, self.matchers["connectives"], strict=self.strict_connectives
            )

        sf_spans: list[Span] = []
        if "sf" in self.matchers:
            sf_spans = pattern_match(doc, self.nlp, self.matchers["sf"])
            if self.sf_final_filter:
                sf_spans = filter_sf_final_bunsetu(sf_spans, doc)

        # Invalidate overlapping connective and sentence-final matches
        connectives_to_remove = set()
        sf_to_remove = set()
        if connective_spans and sf_spans:
            for c_span in connective_spans:
                for sf_span in sf_spans:
                    if (
                        c_span.start_char < sf_span.end_char
                        and sf_span.start_char < c_span.end_char
                    ):
                        connectives_to_remove.add(c_span)
                        sf_to_remove.add(sf_span)

        final_connective_spans = [
            s for s in connective_spans if s not in connectives_to_remove
        ]
        final_sf_spans = [s for s in sf_spans if s not in sf_to_remove]

        for span in final_connective_spans:
            cat0 = connectives_classifications.get(span.label_, "")
            cat1 = cat0
            out.append(
                DmMatch(
                    span=span,
                    表現=span.label_,
                    タイプ="接続表現",
                    機能=cat0,
                    細分類=cat1,
                    position=span.start_char / len(doc.text) if doc.text else 0.0,
                    ジャンル=genre,
                    title=title,
                    sentence_id=sid,
                    section=section,
                )
            )

        for span in final_sf_spans:
            cat0, cat1 = sf_classifications.get(span.label_, ["", ""])
            out.append(
                DmMatch(
                    span=span,
                    表現=span.label_,
                    タイプ="文末表現",
                    機能=cat0,
                    細分類=cat1,
                    position=span.start_char / len(doc.text) if doc.text else 0.0,
                    ジャンル=genre,
                    title=title,
                    sentence_id=sid,
                    section=section,
                )
            )

        doc._.dm_matches = out
        return doc
