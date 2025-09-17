import orjson
import pytest
import spacy
from typer.testing import CliRunner

from dm_annotations.cli import app
from dm_annotations.io.corpus import parse_docs
from dm_annotations.io.export import export_dms


@pytest.fixture
def nlp_with_extractor(model):
    """Create nlp with dm_extractor properly configured."""
    import sys

    spacy = sys.modules["spacy"]
    nlp = spacy.load(model, disable=["ner"])

    # Verify factory is registered - fix the check
    if "dm_extractor" not in nlp.factory_names:
        raise RuntimeError(
            f"dm_extractor factory not registered. Available: {list(nlp.factory_names)}"
        )

    # Add the component
    if "dm_extractor" not in nlp.pipe_names:
        nlp.add_pipe("dm_extractor", last=True)

    # Verify the component was added and has matchers
    if "dm_extractor" not in nlp.pipe_names:
        raise RuntimeError(f"dm_extractor not in pipeline: {nlp.pipe_names}")

    extractor = nlp.get_pipe("dm_extractor")
    if not hasattr(extractor, "matchers") or not extractor.matchers:
        raise RuntimeError("DmExtractor has no matchers")

    print(f"DmExtractor created with matchers: {list(extractor.matchers.keys())}")

    return nlp


@pytest.fixture
def test_texts():
    """Texts that should definitely match our patterns."""
    return [
        "その結果、これは正しいと思う。",  # その結果 (connective) + と思う (SF)
        "しかし、問題があるだろう。",  # しかし (connective) + だろう (SF)
        "さらに、実験は成功したのだ。",  # さらに (connective) + のだ (SF)
        "だから、結論は明らかである。",  # だから (connective) + である (SF)
        "また、分析結果は興味深い。",  # また (connective)
        "これは重要なことだ。",  # だ (SF)
        "研究は完了したのである。",  # のである (SF)
    ]


def test_dm_extractor_component_directly(nlp_with_extractor, test_texts):
    """Test the DmExtractor component directly with known matching texts."""
    nlp = nlp_with_extractor

    for text in test_texts:
        # Process with full pipeline
        doc = nlp(text)

        # Set metadata BEFORE running extractor
        doc.user_data["meta"] = {
            "title": "test",
            "genre": "test",
            "sentence_id": 0,
            "basename": "test",
            "author": "test_author",
            "year": 2024,
            "paragraph_id": None,
            "section": None,
        }

        # Re-run extractor with metadata available
        extractor = nlp.get_pipe("dm_extractor")
        doc = extractor(doc)

        print(f"Text: {text}")
        print(f"Tokens: {[(t.text, t.pos_, t.tag_) for t in doc]}")
        print(f"DM matches: {len(doc._.dm_matches)}")
        for match in doc._.dm_matches:
            print(f"  - {match['表現']} ({match['タイプ']}) at {match['span'].text}")

        # Should find at least one match in these carefully chosen texts
        assert len(doc._.dm_matches) > 0, f"Expected matches in: {text}"


def test_matcher_components_separately(nlp_with_extractor):
    """Test connectives and SF matchers separately."""
    from dm_annotations.pipeline.matcher import (
        connectives_match,
        create_connectives_matcher,
        create_sf_matcher,
        pattern_match,
    )

    nlp = nlp_with_extractor

    # Test connectives matcher
    _, conn_matcher = create_connectives_matcher(nlp)
    conn_text = "その結果、これは正しい。"
    conn_doc = nlp(conn_text)
    conn_matches = connectives_match(conn_doc, nlp, conn_matcher)

    print(f"Connective text: {conn_text}")
    print(f"Connective matches: {[(m.text, m.label_) for m in conn_matches]}")
    assert len(conn_matches) > 0, "Should find 'その結果' connective"

    # Test SF matcher
    _, sf_matcher = create_sf_matcher(nlp)
    sf_text = "これは正しいと思う。"
    sf_doc = nlp(sf_text)
    sf_matches = pattern_match(sf_doc, nlp, sf_matcher)

    print(f"SF text: {sf_text}")
    print(f"SF matches: {[(m.text, m.label_) for m in sf_matches]}")
    assert len(sf_matches) > 0, "Should find 'と思う' or similar SF pattern"


@pytest.fixture
def tmp_jsonl_with_good_texts(tmp_path, test_texts):
    """Create JSONL with texts that should match patterns."""
    p = tmp_path / "good_texts.jsonl"
    rec = {"title": "Test Document", "genre": ["test"], "sentences": test_texts}
    with open(p, "wb") as f:
        f.write(orjson.dumps(rec, option=orjson.OPT_APPEND_NEWLINE))
    return p


def test_full_pipeline_with_good_texts(tmp_jsonl_with_good_texts, model):
    """Test full pipeline with texts that should definitely match."""
    import dm_annotations.pipeline.pipeline  # noqa: F401

    nlp = spacy.load(model, disable=["ner"])
    nlp.add_pipe("dm_extractor", last=True)

    docs = list(parse_docs(tmp_jsonl_with_good_texts, nlp, batch_size=10))

    print(f"Processed {len(docs)} docs")
    for i, doc in enumerate(docs):
        print(f"Doc {i}: '{doc.text}' -> {len(doc._.dm_matches)} matches")
        for match in doc._.dm_matches:
            print(f"  - {match['表現']} ({match['タイプ']})")

    # Should have multiple docs with matches
    docs_with_matches = [doc for doc in docs if len(doc._.dm_matches) > 0]
    assert len(docs_with_matches) > 0, "Expected at least some docs with DM matches"

    # Test export
    all_matches = [doc._.dm_matches for doc in docs]
    out_csv = tmp_jsonl_with_good_texts.parent / "test_out.csv"
    export_dms(all_matches, str(out_csv))
    assert out_csv.exists()

    # Check CSV has content
    with open(out_csv, "r") as f:
        lines = f.readlines()
        assert len(lines) > 1, "CSV should have header + data rows"


def test_corpus_parser_with_iterator(nlp_with_extractor, test_texts):
    """Test CorpusParser with iterator input (like directory processing)."""
    from dm_annotations.loader import CorpusParser

    # Create iterator of (text, metadata) tuples
    sentence_tuples = [
        (
            text,
            {
                "title": "Test Doc",
                "genre": ["test"],
                "basename": "test",
                "author": "Test Author",
                "year": 2024,
                "sentence_id": i,
                "paragraph_id": None,
                "section": None,
            },
        )
        for i, text in enumerate(test_texts)
    ]

    parser = CorpusParser(iter(sentence_tuples), nlp_with_extractor)
    docs = list(parser.stream())

    print(f"CorpusParser processed {len(docs)} docs")
    for doc in docs:
        print(f"'{doc.text}' -> {len(doc._.dm_matches)} matches")
        print(f"Metadata: {doc.user_data.get('meta', {})}")

    # Should have docs with matches
    total_matches = sum(len(doc._.dm_matches) for doc in docs)
    assert total_matches > 0, f"Expected DM matches, got {total_matches}"


def test_pattern_definitions_are_valid(nlp_with_extractor):
    """Test that our pattern definitions are valid and can match."""
    from spacy.matcher import Matcher

    from dm_annotations.pipeline.patterns import connectives_patterns, sf_patterns

    nlp = nlp_with_extractor

    # Test connectives patterns
    conn_matcher = Matcher(nlp.vocab, validate=True)
    for pattern in connectives_patterns:
        try:
            conn_matcher.add(pattern["conjunction"], pattern["pattern"])
        except Exception as e:
            pytest.fail(f"Invalid connective pattern {pattern['conjunction']}: {e}")

    # Test SF patterns
    sf_matcher = Matcher(nlp.vocab, validate=True)
    for name, pattern_def in sf_patterns.items():
        try:
            sf_matcher.add(name, pattern_def["pattern"])
        except Exception as e:
            pytest.fail(f"Invalid SF pattern {name}: {e}")

    print(f"Loaded {len(conn_matcher)} connective patterns")
    print(f"Loaded {len(sf_matcher)} SF patterns")


def test_cli_extract_with_good_data(tmp_jsonl_with_good_texts, model):
    """Test CLI extract command with data that should produce results."""
    out_csv = tmp_jsonl_with_good_texts.parent / "cli_test.csv"
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "extract",
            str(tmp_jsonl_with_good_texts),
            str(out_csv),
            "--model",
            model,
            "--batch-size",
            "10",
            "-vv",  # Debug logging
        ],
    )

    print("CLI stdout:", result.stdout)
    print("CLI stderr:", result.stderr)

    assert result.exit_code == 0, f"CLI failed: {result.stdout}"
    assert out_csv.exists(), "Output CSV not created"

    # Check CSV has actual data
    with open(out_csv, "r") as f:
        lines = f.readlines()
        print(f"CSV has {len(lines)} lines")
        if len(lines) > 1:
            print("Sample CSV line:", lines[1])
        assert len(lines) > 1, "CSV should have data beyond header"


@pytest.mark.parametrize(
    "text,expected_type",
    [
        ("その結果、正しい。", "接続表現"),
        ("これは重要だ。", "文末表現"),
        ("しかし、問題がある。", "接続表現"),
        ("実験は成功したのである。", "文末表現"),
    ],
)
def test_specific_pattern_matches(nlp_with_extractor, text, expected_type):
    """Test specific texts that should match specific pattern types."""
    doc = nlp_with_extractor(text)

    # Set metadata
    doc.user_data["meta"] = {
        "title": "test",
        "genre": "test",
        "sentence_id": 0,
        "basename": "test",
        "author": "test",
        "year": 2024,
        "paragraph_id": None,
        "section": None,
    }

    # Re-run extractor
    extractor = nlp_with_extractor.get_pipe("dm_extractor")
    doc = extractor(doc)

    matches = doc._.dm_matches
    print(f"Text: {text}")
    print(f"Matches: {[(m['表現'], m['タイプ']) for m in matches]}")

    assert len(matches) > 0, f"No matches found for: {text}"

    # Check if we found the expected type
    found_types = [m["タイプ"] for m in matches]
    assert expected_type in found_types, (
        f"Expected {expected_type}, found {found_types}"
    )


def test_debug_pattern_loading_detailed(nlp_with_extractor):
    """Detailed debug of pattern loading and matching."""
    from dm_annotations.pipeline.matcher import create_sf_matcher, pattern_match
    from dm_annotations.pipeline.patterns import connectives_patterns, sf_patterns

    print(f"Available SF patterns: {list(sf_patterns.keys())}")
    print(
        f"Available connective patterns: {[p['conjunction'] for p in connectives_patterns]}"
    )

    # Check if "だ" pattern exists and what it looks like
    if "だ" in sf_patterns:
        print(f"だ pattern: {sf_patterns['だ']}")
    else:
        print("だ pattern not found!")
        print(
            f"Available patterns starting with だ: {[k for k in sf_patterns.keys() if k.startswith('だ')]}"
        )

    # Test the matcher creation
    nlp = nlp_with_extractor
    try:
        _, sf_matcher = create_sf_matcher(nlp)
        print(f"SF matcher created successfully with {len(sf_matcher)} patterns")

        # Test if "だ" is in the matcher
        if "だ" in sf_matcher:
            print("だ pattern is in matcher")
        else:
            print("だ pattern NOT in matcher")
            print(f"Matcher contains: {list(sf_matcher)}")
    except Exception as e:
        print(f"Error creating SF matcher: {e}")

    # Test direct pattern matching
    text = "これは重要だ。"
    doc = nlp(text)
    print(f"Text: {text}")
    print(f"Tokens: {[(t.text, t.pos_, t.tag_, t.lemma_, t.norm_) for t in doc]}")

    try:
        matches = pattern_match(doc, nlp, sf_matcher)
        print(f"Pattern matches: {[(m.text, m.label_) for m in matches]}")
    except Exception as e:
        print(f"Error in pattern matching: {e}")


def test_simple_end_to_end():
    """Test the simplest possible case to verify everything works."""
    import spacy

    import dm_annotations.pipeline.pipeline  # noqa: F401

    # Create fresh nlp instance
    nlp = spacy.load("ja_ginza", disable=["ner"])

    # Add component with sf_final_filter disabled for this test
    nlp.add_pipe("dm_extractor", last=True, config={"sf_final_filter": False})

    # Verify it's there
    assert "dm_extractor" in nlp.pipe_names

    # Test with very simple text
    doc = nlp("なお、花子は街に行くことにしたという。")
    # Run all pipes except dm_extractor to ensure POS and parser are applied
    with nlp.select_pipes(disable=["dm_extractor"]):
        doc = nlp(doc.text)
    doc.user_data["meta"] = {
        "title": "test",
        "genre": "test",
        "sentence_id": 0,
        "basename": "test",
        "author": "test",
        "year": 2024,
        "paragraph_id": None,
        "section": None,
    }
    # Re-run extractor
    extractor = nlp.get_pipe("dm_extractor")
    doc = extractor(doc)

    print(f"Text: {doc.text}")
    print(f"Tokens: {[(t.text, t.pos_, t.tag_, t.lemma_) for t in doc]}")
    print(f"Matches: {len(doc._.dm_matches)}")
    for match in doc._.dm_matches:
        print(f"  Match: {match}")

    # This should definitely work
    assert len(doc._.dm_matches) > 1, "Should find >= 2 pattern"
