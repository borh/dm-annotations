import logging

import polars as pl
import pytest
import spacy

from dm_annotations.io.export import export_dms
from dm_annotations.io.loader import CorpusParser
from dm_annotations.io.text_loader import (
    _normalize_section_name,
    parse_plain_folder_to_tuples,
)


# 1. Unit-test _normalize_section_name behaviour
@pytest.mark.parametrize(
    "heading,expected",
    [
        ("はじめに", "introduction"),
        ("結論", "conclusion"),
        ("方法", "methods"),
        ("参考文献", "references"),
        ("謝辞", "acknowledgments"),
        ("キーワード", "keywords"),
    ],
)
def test_normalize_section_name_matches(heading, expected):
    assert _normalize_section_name(heading) == expected


# 2. Integration: parse_plain_folder_to_tuples yields per-segment section metadata
def test_section_tagging_and_logging(tmp_path, caplog):
    # Temporary folder structure with metadata TSVs
    content = """basename\tjournal_title\tauthor\tjournal_name\tvol\tno\tページ数\turl
testdoc\tTest Journal\tTester\tTest Pub\t1\t1\t1-5\thttp://fake/2024/
"""
    (tmp_path / "sources.tsv").write_text(content, encoding="utf-8")
    (tmp_path / "genres.tsv").write_text("CODE\tTestGenre\n", encoding="utf-8")

    # Create simple markdown with headings & text
    md_text = """# タイトル

## はじめに

これはイントロです。

## 方法

これは方法の説明です。

## 結論

結論の文です。
"""
    (tmp_path / "testdoc.md").write_text(md_text, encoding="utf-8")

    caplog.set_level(logging.DEBUG)
    tuples = list(parse_plain_folder_to_tuples(tmp_path, strict_metadata=True))
    print(tuples)
    # Expect 3 paragraphs with sections assigned
    sections = [meta["section"] for _, meta in tuples]
    print(sections)
    assert any(sec == "introduction" for sec in sections)
    assert any(sec == "methods" for sec in sections)
    assert any(sec == "conclusion" for sec in sections)

    # Check logging includes our section tree lines
    log_output = "\n".join(caplog.messages)
    assert "=== Section Map for testdoc.md" in log_output
    assert "introduction" in log_output
    assert "methods" in log_output
    assert "conclusion" in log_output


# 3. End-to-end: section metadata should survive into dm_matches & CSV
def test_section_propagates_into_matches_and_csv(tmp_path):
    # Reuse the same folder structure
    content = """basename\tjournal_title\tauthor\tjournal_name\tvol\tno\tページ数\turl
testdoc\tTest Journal\tTester\tTest Pub\t1\t1\t1-5\thttp://fake/2024/
"""
    (tmp_path / "sources.tsv").write_text(content, encoding="utf-8")
    (tmp_path / "genres.tsv").write_text("CODE\tTestGenre\n", encoding="utf-8")

    md_text = """# タイトル
## はじめに
その結果、これは正しいと思う。

## 結論
これは重要だ。
"""
    (tmp_path / "testdoc.md").write_text(md_text, encoding="utf-8")

    # Pipe: tuples -> CorpusParser -> DmExtractor
    nlp = spacy.load("ja_ginza", disable=["ner"])
    nlp.add_pipe("dm_extractor", last=True)
    tuples = parse_plain_folder_to_tuples(tmp_path)
    parser = CorpusParser(tuples, nlp)
    docs = list(parser.stream())

    # Ensure that sections propagate through doc._.dm_matches
    for doc in docs:
        for match in doc._.dm_matches:
            assert "section" in match
            # The section should match one of the normalized names or None
            assert match["section"] in (None, "introduction", "conclusion")

    # Export to CSV and confirm 'section' column exists
    out_csv = tmp_path / "out.csv"
    export_dms((doc._.dm_matches for doc in docs), str(out_csv))
    df = pl.read_csv(out_csv)
    assert "section" in df.columns
    # At least one row with introduction or conclusion section, if any matches exist
    if len(df) > 0:
        assert df["section"].is_in(["introduction", "conclusion"]).any()
