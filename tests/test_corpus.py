from pathlib import Path
from logging import warn
import spacy
from dm_annotations.corpus import parse_or_get_docs, extract_dm


def test_parse_or_get_docs(model):
    path = Path("resources/test.jsonl")
    nlp = spacy.load(model)
    docs = parse_or_get_docs(path, nlp, batch_size=1)
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

    dms = list(extract_dm(docs, nlp))
    print(docs, dms)
    assert len(list(matches for matches in dms if matches)) > 0
