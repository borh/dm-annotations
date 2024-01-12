import logging
from pathlib import Path
import re

import jaconv
import numpy as np
import orjson
import pandas as pd
from prodigy import set_hashes
from prodigy.components.db import connect
import spacy
from spacy import displacy
from spacy.tokens import Doc, Span, SpanGroup

from dm_annotations.matcher import (
    connectives_match,
    create_connectives_matcher,
    create_modality_matcher,
    pattern_match,
)


if not Doc.has_extension("section_name"):
    Doc.set_extension("section_name", default=None)

if not Doc.has_extension("title"):
    Doc.set_extension("title", default=None)

if not Doc.has_extension("genre"):
    Doc.set_extension("genre", default=None)

if not Doc.has_extension("sentence_id"):
    Doc.set_extension("sentence_id", default=None)

if not Doc.has_extension("segment_id"):
    Doc.set_extension("segment_id", default=None)


def normalize_nfkc(s):
    if isinstance(s, str):
        return jaconv.h2z(
            jaconv.normalize(s.replace(" ", ""), "NFKC"),
            digit=False,
            ascii=True,
            kana=True,
        ).rstrip("　")
    return s


def add_annotation(doc, span, key="sc"):
    if span is not None:
        if key in doc.spans:
            doc.spans[key] += [span]
        else:
            doc.spans[key] = [span]


def add_annotations(doc, spans, key="sc"):
    if spans is not None:
        if key in doc.spans:
            doc.spans[key].extend(spans)
        else:
            doc.spans[key] = spans


def add_regex_span(doc, regex, label, key="sc", meidai=False, reverse_find=False):
    if not isinstance(regex, str):
        return None
    regex = re.sub(r"[／（）。．？]", "", regex)
    if not regex:
        return None
    try:
        if regex[-1] == "，" or regex[-1] == "、":
            regex = regex[:-1]
        rx = re.compile(regex)
    except re.error as e:
        logging.error(f"'{regex}' not valid regex: {e}")
        return None
    matches = list(rx.finditer(doc.text))
    if reverse_find:
        matches = matches[::-1]
    for match in matches:
        start, end = match.span()
        span = doc.char_span(start, end, label=label)
        add_annotation(doc, span, key=key)
        return None  # Only look at start/end (reverse_find) of segment
    logging.error(f"'{regex}' not found in '{doc}'!")


def read_annotations_excel(path, model="ja_ginza"):
    df = pd.read_excel(
        path,
    )
    # Apply NFKC normalization on string columns
    string_columns = df.select_dtypes(include=["object"]).columns
    df[string_columns] = df[string_columns].map(normalize_nfkc)

    # 項目番号	節の名称	文段番号章‗節＿段（７章）	文番号(186)題目含む	segment数（221）	テクスト単位（文単位）(15882文字）	segment文章（アンダーラインは命題部分）	文構造	文頭句	接続表現（５２３項目含まれる１　外０）	命題の内外（外１/内０）	備考	文末表現	基本形	文末表現意味機能分類（現表にない場合は０）	命題の内か外か	備考

    rename_dict = {
        0: "_",
        1: "section_name",
        2: "page_index",
        3: "rid",
        4: "segments",
        5: "sentence",
        6: "segment",
        7: "dm",
        8: "connective",
        9: "known_connective",
        10: "connective_meidai_check",
        11: "connective_info",
        12: "modality",
        13: "modality_normal_form",
        14: "modality_bunrui",
        15: "modality_meidai_check",
        16: "modality_info",
    }
    df.columns = [rename_dict.get(i, col) for i, col in enumerate(df.columns)]

    bool_map = {1: True, 0: False, "": np.nan, "1": True, "0": False}
    # False (0): Inside meidai, True(1): Outside meidai
    df["connective_meidai_check"] = df["connective_meidai_check"].map(bool_map)
    df["modality_meidai_check"] = df["modality_meidai_check"].map(bool_map)
    df["dm"] = df["dm"].map(str)
    df["rid"] = df["rid"].map(str).map(lambda s: jaconv.z2h(s, ascii=True, digit=True))
    # df = df.map(
    #     lambda s: jaconv.h2z(
    #         jaconv.normalize(s, "NFKC"), digit=False, ascii=True, kana=True
    #     )
    #     if isinstance(s, str)
    #     else s,
    #     na_action="ignore",
    # )
    df.fillna("", inplace=True)
    df["segment"] = df["segment"].apply(normalize_nfkc)
    df["sentence"] = df["sentence"].apply(normalize_nfkc)
    df["connective"] = df["connective"].apply(normalize_nfkc)
    df["modality"] = df["modality"].apply(normalize_nfkc)

    nlp = spacy.load(model)
    nlp, modality_matcher = create_modality_matcher(nlp=nlp)
    nlp, connectives_matcher = create_connectives_matcher(nlp=nlp)
    P_RX = re.compile(r"S0*(?P<sentence>\d{1,3})")
    DM_FRAGMENT = r"(\w+)\([\)]+\)"
    DM_RX = re.compile(rf"(\[?(((\w+)\([^\)]+\))+?)\]?)+?")

    docs = []
    sentence_id, segment, section_name, title = None, -1, None, None
    for r in df.to_dict("records"):
        rid = r["rid"]
        dm_pattern = r["dm"]
        new_section_name = r["section_name"]
        if new_section_name:
            section_name = new_section_name
        if title:
            pass
        elif section_name == "タイトル":
            title = r["sentence"]
        elif r["segments"] == "タイトル" and r["sentence"]:
            title = r["sentence"]
        # print(rid, "=" * 20, dm_pattern, DM_RX.match(dm_pattern), section_name)

        if match := P_RX.match(rid):
            new_sentence_id = int(match.group("sentence"))

            if new_sentence_id != sentence_id:
                sentence_id = new_sentence_id
                segment = 1
            else:
                segment += 1
        else:
            # Same sentence_id, new segment
            if segment:
                segment += 1
            else:
                segment = 1

        if r["segment"]:
            doc = nlp(r["segment"].rstrip())
        elif r["sentence"]:
            doc = nlp(r["sentence"].rstrip())
        else:
            continue

        doc._.section_name = section_name
        doc._.sentence_id = sentence_id
        doc._.segment_id = segment
        doc._.title = title
        add_regex_span(
            doc,
            r["connective"],
            "c",
            key="c",
            meidai=r["connective_meidai_check"],
        )
        add_regex_span(
            doc,
            r["modality"],
            "sf",
            key="sf",
            meidai=r["modality_meidai_check"],
        )

        if modality_matches := pattern_match(doc, nlp, modality_matcher):
            add_annotations(doc, modality_matches, key="m_auto")
        if connectives_matches := connectives_match(doc, nlp, connectives_matcher):
            add_annotations(doc, connectives_matches, key="c_auto")
        segment_span = Span(doc, 0, len(doc), label=f"{sentence_id}_segment_{segment}")
        add_annotation(doc, segment_span, key="segment")
        add_annotation(
            doc,
            Span(doc, 0, len(doc), label=section_name),
            key="section_name",
        )

        docs.append(doc)

        assert 0 <= sentence_id - docs[-1]._.sentence_id <= 1

    docs = merge_docs(nlp, docs)

    return df, docs


def merge_contiguous_spans(doc, spans):
    merged_spans = []

    for span in spans:
        # Check if there are no merged spans yet or if the current span's label differs from the last merged span's label.
        if not merged_spans or span.label_ != merged_spans[-1].label_:
            # Add the current span as a new merged span.
            merged_spans.append(span)
        else:
            # Extend the last merged span to include the current span.

            merged_spans[-1] = Span(
                doc, merged_spans[-1].start, span.end, label=span.label_
            )

    assert spans != merged_spans

    return merged_spans


def merge_docs(nlp, docs):
    # Create new Doc
    pretty_text = ""
    for doc in docs:
        # if doc._.segment_id == 1:
        #    pretty_text += "\n"
        pretty_text += doc.text

    new_doc = nlp(pretty_text)
    print(len(list(new_doc.sents)))

    # Copy custom attributes
    # First set Doc attributes using only the first doc passed in
    for custom_attr in docs[0]._._extensions.keys():
        new_doc._.set(custom_attr, docs[0]._.get(custom_attr))
    assert new_doc._._extensions.keys()
    assert new_doc._.title
    # Merge all span attributes from docs, taking care to fix offsets and recreating from character indexes and not token indexes, as they may have changed with the new_doc tokenisation.
    char_offset = 0
    for doc in docs:
        for custom_attr in doc.spans.keys():
            spans = [
                new_doc.char_span(
                    s.start_char + char_offset,
                    s.end_char + char_offset,
                    s.label,
                    alignment_mode="expand",  # TODO check
                )
                for s in doc.spans.get(custom_attr)
            ]

            add_annotations(new_doc, spans, custom_attr)

        char_offset += len(doc.text)

    new_doc.spans["section_name"] = merge_contiguous_spans(
        new_doc, new_doc.spans["section_name"]
    )

    return new_doc


def spacy_to_prodigy_spans(doc):
    spans = []
    for custom_attr in doc.spans.keys():
        spans.extend(
            [
                {
                    "start": s.start_char,
                    "end": s.end_char,
                    "label": s.label_,
                }
                for s in doc.spans.get(custom_attr)
            ]
        )
    return spans


if __name__ == "__main__":
    with open("annotations.jsonl", "wb") as f:
        db = connect()
        examples = []
        for excel in Path("resources/analyses/").glob("*.xlsx"):
            print(excel)
            df, doc = read_annotations_excel(excel, model="ja_ginza")
            assert len(df) > 0
            assert len(doc) > 0

            assert {"section_name", "rid", "dm"} <= set(df.columns.to_list())

            f.write(orjson.dumps(doc.to_json(), option=orjson.OPT_APPEND_NEWLINE))

            spans = [
                {"start": span.start_char, "end": span.end_char, "label": span.label_}
                for span_type in doc.spans.keys()
                for span in doc.spans.get(span_type, [])
            ]
            examples.append(
                {
                    "text": doc.text,
                    "spans": spans,
                    "title": doc._.title,
                    "answer": "accept",
                }
            )

            Path(f"annotations_{doc._.title}.html").unlink(missing_ok=True)
            Path(f"annotations_deps_{doc._.title}.html").unlink(missing_ok=True)
            # for doc in doc:
            colors = [
                "#F8766D",
                "#CD9600",
                "#7CAE00",
                "#00BE67",
                "#00BFC4",
                "#00A9FF",
                "#C77CFF",
                "#FF61CC",
            ]
            color_map = {
                span.label_: colors[0] for span in doc.spans.get("c", SpanGroup(doc))
            }
            color_map |= {
                span.label_: colors[1] for span in doc.spans.get("sf", SpanGroup(doc))
            }
            color_map |= {
                span.label_: colors[2]
                for span in doc.spans.get("segment", SpanGroup(doc))
            }
            color_map |= {
                span.label_: colors[3]
                for span in doc.spans.get("section_name", SpanGroup(doc))
            }
            doc.spans["sc"] = (
                doc.spans.get("c", SpanGroup(doc))
                + doc.spans.get("sf", SpanGroup(doc))
                + doc.spans.get("segment", SpanGroup(doc))
                + doc.spans.get("section_name", SpanGroup(doc))
            )

            svg = displacy.render(
                doc,
                style="span",
                options={"colors": color_map},
            )
            deps_svg = displacy.render(
                doc.sents, style="dep", options={"compact": True}
            )
            Path(f"annotations_{doc._.title}.html").open("w").write(svg)
            Path(f"annotations_{doc._.title}_deps.html").open("w").write(deps_svg)

        db.add_dataset("imported_annotations")  # add dataset ner_sample
        examples = (set_hashes(eg) for eg in examples)  # add hashes; creates generator
        db.add_examples(
            list(examples), ["imported_annotations"]
        )  # add examples to ner_sample; need list as was generator
    # colors = [
    #     "#F8766D",
    #     "#CD9600",
    #     "#7CAE00",
    #     "#00BE67",
    #     "#00BFC4",
    #     "#00A9FF",
    #     "#C77CFF",
    #     "#FF61CC",
    # ]

    # nlp = spacy.load("ja_core_news_trf")
    # docs = nlp.pipe(text.splitlines())
    # Path("ryu.html").unlink(missing_ok=True)
    # Path("ryu_deps.html").unlink(missing_ok=True)
    # for doc in docs:
    #     m_spans = modality_match(doc)
    #     c_spans = connectives_match(doc)

    #     doc.spans["modality"] = m_spans
    #     doc.spans["connectives"] = c_spans
    #     doc.spans["sc"] = m_spans + c_spans

    #     color_map = {span.label_: colors[0] for span in m_spans}
    #     color_map |= {span.label_: colors[1] for span in c_spans}

    #     svg = displacy.render(doc, style="span", options={"colors": color_map})
    #     deps_svg = displacy.render(doc, style="dep", options={"compact": True})
    #     Path("ryu.html").open("a").write(svg)
    #     Path("ryu_deps.html").open("a").write(deps_svg)
    # for excel in Path("resources/analyses/").glob("*.xlsx"):
    #     assert read_annotations_excel(excel) is not None
