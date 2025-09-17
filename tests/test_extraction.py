from spacy.language import Language

from dm_annotations.pipeline.pipeline_components import DmExtractor


def test_connective_extraction_mid_sentence(nlp: Language):
    """
    Test connective extraction on a sentence with a mid-sentence connective.

    This is based on a user report that "あと" was being incorrectly matched
    in a long sentence. The matching logic should only find connectives at
    the beginning of a sentence.
    """
    text = "ねえ、みかんやってみたりしたってのもそうなんだろうしさあ、あとそれから、あれだって開発だなんて期待があったけど、ああ、期待ってほどでもないけど、なんかあるかって思ったりもしたけどねぇ、どうでしょう。"

    # The default DmExtractor uses strict_connectives=False.
    # We create an instance to be sure of the settings.
    extractor = DmExtractor(nlp, strict_connectives=False)
    doc = extractor(nlp(text))
    matches = doc._.dm_matches

    connectives = [m for m in matches if m["タイプ"] == "接続表現"]

    # There should be no match for "あと" because it's not at the start of the sentence.
    ato_matches = [c for c in connectives if c["表現"] == "あと"]
    assert not ato_matches, f"Incorrectly matched 'あと' as a connective: {ato_matches}"

    # The only possible match is at the start of the sentence.
    # The sentence starts with 'ねえ'. If 'ねえ' is a connective, it should be the only one.
    # If not, no connectives should be found.
    if connectives:
        assert (
            len(connectives) == 1
        ), "Should only find at most one connective at the start of the sentence"
        # The only possible match is at the start.
        assert (
            connectives[0]["span"].start_char == 0
        ), "Connective match must be at the start of the sentence"
