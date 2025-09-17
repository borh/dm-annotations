from pathlib import Path

import spacy

from dm_annotations.io.corpus import parse_or_get_docs


def test_parse_or_get_docs(model):
    path = Path("resources/test.jsonl")
    spacy.prefer_gpu()
    nlp = spacy.load(model, disable=["ner"])
    # register our DM‐extractor pipe
    nlp.add_pipe("dm_extractor", last=True)
    docs = list(parse_or_get_docs(path, nlp, batch_size=1))
    assert docs

    assert_docs = list(
        nlp.pipe(
            [
                "この世に及んで、「魅力」は概念として随分語られてきたでしょう。",
                "魅力は学術的な観点から1890年代を持って大きく研究されるようになった．",
                "その成果として，「魅力の効果」が解明に寄与したことがいえるであろう．",
            ]
        )
    )
    # Test initial load and cache:
    assert [doc.text for doc in docs] == [doc.text for doc in assert_docs]
    # Test cached results:
    docs = list(parse_or_get_docs(path, nlp, batch_size=1))
    assert [doc.text for doc in docs] == [doc.text for doc in assert_docs]

    # flatten all doc._.dm_matches
    dms = [m for doc in docs for m in doc._.dm_matches]
    assert dms, "expected at least one DM match"
