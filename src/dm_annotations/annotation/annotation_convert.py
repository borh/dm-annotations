"""
Convert a normalized DataFrame into a list of spaCy Doc objects with spans.
"""

import re
from pathlib import Path
from typing import List

import numpy as np
import polars as pl
from spacy.tokens import Doc, Span

from dm_annotations.annotation.annotation_io import read_annotations
from dm_annotations.pipeline.matcher import (
    connectives_match,
    create_connectives_matcher,
    create_modality_matcher,
    pattern_match,
)


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
        print(f"'{regex}' not valid regex: {e}")
        return None
    matches = list(rx.finditer(doc.text))
    if reverse_find:
        matches = matches[::-1]
    for match in matches:
        start, end = match.span()
        span = doc.char_span(start, end, label=label)
        add_annotation(doc, span, key=key)
        return None  # Only look at start/end (reverse_find) of segment
    print(f"'{regex}' not found in '{doc}'!")


def merge_contiguous_spans(doc, spans):
    from spacy.tokens import Span

    merged_spans: list[Span] = []
    for span in spans:
        if not merged_spans or span.label_ != merged_spans[-1].label_:
            merged_spans.append(span)
        else:
            merged_spans[-1] = Span(
                doc, merged_spans[-1].start, span.end, label=span.label_
            )
    assert spans != merged_spans
    return merged_spans


def merge_docs(nlp, docs):
    pretty_text = ""
    for doc in docs:
        pretty_text += doc.text

    new_doc = nlp(pretty_text)

    # Copy custom attributes from first doc
    for custom_attr in docs[0]._._extensions.keys():
        new_doc._.set(custom_attr, docs[0]._.get(custom_attr))

    # Merge all span attributes
    char_offset = 0
    for doc in docs:
        for custom_attr in doc.spans.keys():
            spans = [
                new_doc.char_span(
                    s.start_char + char_offset,
                    s.end_char + char_offset,
                    s.label,
                    alignment_mode="expand",
                )
                for s in doc.spans.get(custom_attr)
            ]
            add_annotations(new_doc, spans, custom_attr)
        char_offset += len(doc.text)

    new_doc.spans["section_name"] = merge_contiguous_spans(
        new_doc, new_doc.spans["section_name"]
    )
    return new_doc


def convert_annotations(path: Path, model: str = "ja_ginza") -> List[Doc]:
    """
    Load annotations and produce spaCy Doc objects with c/sf spans.
    """
    # Read into Polars DataFrame and normalize columns
    df = read_annotations(path)
    bool_map = {1: True, 0: False, "": np.nan, "1": True, "0": False}
    df = df.with_columns(
        pl.col("connective_meidai_check")
        .map_dict(bool_map)
        .alias("connective_meidai_check"),
        pl.col("modality_meidai_check")
        .map_dict(bool_map)
        .alias("modality_meidai_check"),
        pl.col("dm").cast(pl.Utf8).alias("dm"),
        pl.col("rid").cast(pl.Utf8).alias("rid"),
    )
    # Fill all nulls with empty string
    df = df.fill_null("")

    from dm_annotations import load_core_nlp

    nlp = load_core_nlp(model)
    _, modality_matcher = create_modality_matcher(nlp=nlp)
    _, connectives_matcher = create_connectives_matcher(nlp=nlp)
    P_RX = re.compile(r"S0*(?P<sentence>\d{1,3})")

    docs: list[Doc] = []
    sentence_id: int | None = None
    segment: int = -1
    section_name: str = ""
    title: str = ""

    # iterate over row‐dicts
    for r in df.to_dicts():
        rid = r["rid"]
        new_section_name = r["section_name"]
        if new_section_name:
            section_name = new_section_name
        if title:
            pass
        elif section_name == "タイトル":
            title = r["sentence"]
        elif r["segments"] == "タイトル" and r["sentence"]:
            title = r["sentence"]

        # Parse sentence ID and segment
        if match := P_RX.match(rid):
            new_sentence_id = int(match.group("sentence"))
            if new_sentence_id != sentence_id:
                sentence_id = new_sentence_id
                segment = 1
            else:
                segment += 1
        else:
            if segment:
                segment += 1
            else:
                segment = 1

        # Create doc from segment or sentence
        if r["segment"]:
            doc = nlp(r["segment"].rstrip())
        elif r["sentence"]:
            doc = nlp(r["sentence"].rstrip())
        else:
            continue

        # Store metadata in user_data instead of extensions
        if "meta" not in doc.user_data:
            doc.user_data["meta"] = {}
        doc.user_data["meta"]["section_name"] = section_name
        doc.user_data["meta"]["sentence_id"] = sentence_id
        doc.user_data["meta"]["segment_id"] = segment
        doc.user_data["meta"]["title"] = title

        # Add spans
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

        # Add auto-detected spans
        if modality_matches := pattern_match(doc, nlp, modality_matcher):
            add_annotations(doc, modality_matches, key="m_auto")
        if connectives_matches := connectives_match(doc, nlp, connectives_matcher):
            add_annotations(doc, connectives_matches, key="c_auto")

        # Add segment and section spans
        segment_span = Span(doc, 0, len(doc), label=f"{sentence_id}_segment_{segment}")
        add_annotation(doc, segment_span, key="segment")
        # only produce a section‐span if we have a non‐empty label
        if section_name:
            add_annotation(
                doc,
                Span(doc, 0, len(doc), label=section_name),
                key="section_name",
            )

        docs.append(doc)
        assert 0 <= sentence_id - docs[-1]._.sentence_id <= 1

    # Merge all docs into one
    merged_doc = merge_docs(nlp, docs)
    return [merged_doc]
