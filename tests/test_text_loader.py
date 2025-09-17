from pathlib import Path

import pytest

from dm_annotations.io.text_loader import (
    _extract_text_from_blocks,
    _normalize_section_name,
    _process_md_file,
    parse_md_to_section_nodes,
    remove_cjk_whitespace,
)


def test_normalize_section_name_multiple_matches():
    cats, prefix, _matched = _normalize_section_name("3. 研究の仮説と目的")
    assert isinstance(cats, set)
    assert "hypotheses" in cats
    assert "introduction" in cats
    assert prefix == "3."


def test_propagate_child_categories_and_primary(tmp_path):
    md = """## 3. 研究の仮説と目的
### 3.2. 本研究における仮説
仮説 1: これはテストです。

### 3.3. 研究の目的
目的はテストである。

## 6. 考察
本文。

### 6.3. 今後の研究課題
将来の課題について。
"""
    p = tmp_path / "sample.md"
    p.write_text(md, encoding="utf-8")
    # call the internal worker to get (text, meta) tuples and/or nodes
    info = {
        "basename": "sample",
        "journal_title": "",
        "title": "",
        "author": "",
        "year": 2020,
        "genre": None,
    }
    results = _process_md_file((p, info, set()))
    # also get nodes
    nodes = parse_md_to_section_nodes(p)
    # find the top-level "3. 研究の仮説と目的"
    node3 = next(n for n in nodes if n.raw_text.startswith("3. 研究の仮説と目的"))
    assert isinstance(node3.category, set)
    assert {"hypotheses", "introduction"} <= node3.category
    # find the top-level "6. 考察" and ensure future_work was propagated from child
    node6 = next(n for n in nodes if n.raw_text.startswith("6. 考察"))
    assert isinstance(node6.category, set)
    assert "discussion" in node6.category
    assert "future_work" in node6.category


def test_parse_real_00427_sample(tmp_path):
    # ensure the real sample file exists in repository root tests/assets/ or repo root
    repo_md = Path("00427.md")
    if not repo_md.exists():
        pytest.skip("00427.md sample not available")
    nodes = parse_md_to_section_nodes(repo_md)
    # locate "3. 研究の仮説と目的"
    found = [n for n in nodes if n.raw_text.startswith("3. 研究の仮説と目的")]
    assert found, "expected a node for '3. 研究の仮説と目的'"
    n = found[0]
    assert isinstance(n.category, set)
    assert "hypotheses" in n.category and "introduction" in n.category


def test_header_body_excludes_header_text_00117():
    md = Path("00117.pre.md")
    if not md.exists():
        pytest.skip("00117.pre.md sample not available")
    nodes = parse_md_to_section_nodes(md)
    # find the '統計解析' heading node
    node = next((n for n in nodes if n.raw_text and "統計解析" in n.raw_text), None)
    assert node is not None, "expected a node for '統計解析'"
    header = _extract_text_from_blocks([node.block]).strip()
    header_norm = remove_cjk_whitespace(header)
    assert header_norm not in (node.body_text or ""), (
        f"Header {header_norm!r} found in body_text"
    )


def test_result_section_contains_children_00117():
    md = Path("00117.pre.md")
    if not md.exists():
        pytest.skip("00117.pre.md sample not available")
    nodes = parse_md_to_section_nodes(md)
    # prefer category match, fall back to raw_text search
    node3 = next(
        (n for n in nodes if n.num == 3 and n.category and "results" in n.category),
        None,
    )
    if node3 is None:
        node3 = next(
            (
                n
                for n in nodes
                if n.raw_text
                and n.raw_text.strip().startswith("3")
                and "結果" in n.raw_text
            ),
            None,
        )
    assert node3 is not None, "expected a node for '3 結果'"
    assert any("作業成績" in (c.raw_text or "") for c in node3.children), (
        "Expected child '作業成績' under '3 結果'"
    )


def test_normalize_section_name_single_char_note():
    cats, prefix, matched = _normalize_section_name("注")
    assert isinstance(cats, set)
    assert "notes" in cats
    assert prefix is None
    assert "notes" in (matched or set())


def test_para_plain_single_line_note_heading(tmp_path):
    md = "# Doc Title\n\nこれは本文です。\n\n注\n\n- (1) 注記の内容です。\n"
    p = tmp_path / "note_sample.md"
    p.write_text(md, encoding="utf-8")
    nodes = parse_md_to_section_nodes(p, section_exclude=set())
    # Find a node whose raw_text is the standalone "注" heading
    node = next((n for n in nodes if n.raw_text and n.raw_text.strip() == "注"), None)
    assert node is not None, "expected a node for the standalone '注' paragraph"
    # The node should have matched_subtypes/category that include 'notes'
    found = False
    if getattr(node, "matched_subtypes", None):
        found = found or ("notes" in node.matched_subtypes)
    if getattr(node, "category", None):
        cat = node.category
        if isinstance(cat, (set, list, tuple)):
            found = found or ("notes" in cat)
        else:
            found = found or (cat == "notes")
    assert found, (
        f"'notes' not found in matched_subtypes/category for node: matched={node.matched_subtypes}, category={node.category}"
    )
