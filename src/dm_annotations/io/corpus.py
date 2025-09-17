import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import orjson
import spacy
from spacy.language import Language
from spacy.tokens import Doc

# Removed implicit factory registration — caller must now call register_dm_extractor()
from dm_annotations.io.export import export_count, export_dms
from dm_annotations.io.loader import CorpusParser


def reserialize_corpus_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """
    Group a flat JSONL of single sentences into records of
    { "title": str, "genre": list[str], "sentences": list[str] }.

    >>> from pathlib import Path
    >>> import json, tempfile, os
    >>> # make a temp JSONL
    >>> t = tempfile.NamedTemporaryFile(delete=False, mode="wb")
    >>> t.write(b'{"title":"T","genre":["G"],"sentences":["A"]}\\n')
    >>> t.write(b'{"title":"T","genre":["G"],"sentences":["B"]}\\n')
    >>> t.close()
    >>> out = list(reserialize_corpus_jsonl(Path(t.name)))
    >>> out == [{"title":"T","genre":["G"],"sentences":["A","B"]}]
    True
    >>> os.unlink(t.name)
    """
    title: str = ""
    genre: list[str] = []
    sentences: list[str] = []
    with open(path, "rb") as f:
        for line in f:
            d = orjson.loads(line)
            # if we’ve moved to a new title, emit the last group
            if title and d["title"] != title:
                yield {
                    "title": title,
                    "genre": genre,
                    "sentences": sentences,
                }
                sentences = []
            title = d["title"]
            genre = d["genre"]
            # the incoming JSONL may already have `sentences` or a single `text`
            if "sentences" in d and isinstance(d["sentences"], list):
                sentences.extend(d["sentences"])
            elif "text" in d:
                sentences.append(d["text"])
            else:
                raise KeyError("Each JSONL record needs 'sentences' or 'text'")
    if sentences:
        yield {
            "title": title,
            "genre": genre,
            "sentences": sentences,
        }


def parse_docs(
    path: Path,
    nlp: Language,
    batch_size: int = 100_000,
    pipe_batch_size: int = 800,
    n_process: int = 1,
) -> Iterator[Doc]:
    """
    Lazily parse or load from cache a JSONL at `path` into spaCy Doc objects.

    :param path: path to a JSONL file with {"title","genre","sentences"} groups
    :param nlp: a loaded spaCy Language pipeline
    :param batch_size: number of sentences per cache file
    :param pipe_batch_size: batch size for spaCy .pipe
    :return: iterator of `Doc` with `. _.genre`, `. _.title`, `. _.sentence_id` set

    >>> import spacy
    >>> from pathlib import Path
    >>> nlp = spacy.blank("en")
    >>> list(parse_docs(Path("resources/test.jsonl"), nlp, batch_size=1, pipe_batch_size=1))
    []  # empty test file yields no docs
    """
    return CorpusParser(
        path,
        nlp,
        cache_batch_size=batch_size,
        pipe_batch_size=pipe_batch_size,
        n_process=n_process,
    ).stream()


def parse_or_get_docs(
    path: Path, nlp: Language, batch_size: int = 100_000
) -> Iterator[Doc]:
    """Alias for backward compatibility with existing tests."""
    return parse_docs(path, nlp, batch_size)


if __name__ == "__main__":
    # Import to register factory
    import dm_annotations.pipeline  # noqa: F401

    # Corpus preprocessing
    from dm_annotations.io.loader import CorpusParser

    with open("learner-corpus.jsonl", "wb") as f:
        for text in reserialize_corpus_jsonl(
            Path("resources/learner-2022-09-08.jsonl")
        ):
            f.write(orjson.dumps(text, option=orjson.OPT_APPEND_NEWLINE))
    with open("native-corpus.jsonl", "wb") as f:
        for text in reserialize_corpus_jsonl(Path("resources/native-2022-09-08.jsonl")):
            f.write(orjson.dumps(text, option=orjson.OPT_APPEND_NEWLINE))

    with open("science-corpus.jsonl", "wb") as f:
        for text in reserialize_corpus_jsonl(Path("resources/native-2022-09-08.jsonl")):
            genre = text["genre"][0]
            # filter on science genres
            if genre in {
                "科学技術論文",
                "人文社会学論文",
                # "社会科学専門書",
            }:
                f.write(orjson.dumps(text, option=orjson.OPT_APPEND_NEWLINE))

    # E2E test
    SPACY_MODEL = os.environ.get("SPACY_MODEL", "ja_ginza")
    spacy.prefer_gpu()
    from dm_annotations import load_nlp

    nlp = load_nlp(SPACY_MODEL, disable=["ner"])
    if "ginza" in SPACY_MODEL:
        nlp.add_pipe("disable_sentencizer", before="parser")

    # Add DM pipeline components
    nlp.add_pipe("dm_connectives", last=True)
    nlp.add_pipe("dm_sf", last=True)
    nlp.add_pipe(
        "doc_cleaner", config={"attrs": {"tensor": None}}
    )  # Lower memory usage

    # docs = parse_docs(Path("learner-corpus.jsonl"), nlp)
    # matches = []
    # for doc in nlp.pipe(docs, as_tuples=False, batch_size=PIPE_BATCH_SIZE, n_process=1):
    #     matches.append(doc._.dm_connectives + doc._.dm_sf)
    # print(len(matches), len([match for match in matches if match]))
    # export_dms(matches, "learner-dms.csv")
    # export_count("learner-dms.csv")

    docs = parse_docs(Path("science-corpus.jsonl"), nlp)
    matches = []
    for i, doc in enumerate(docs):
        doc._.sentence_id = i
        processed_doc = nlp(doc.text)
        processed_doc._.genre = doc._.genre
        processed_doc._.title = doc._.title
        processed_doc._.sentence_id = doc._.sentence_id
        linear_matches = sorted(
            processed_doc._.dm_connectives + processed_doc._.dm_sf,
            key=lambda d: d["span"].start_char,
        )
        matches.append(linear_matches)
    print(len(matches), len([match for match in matches if match]))
    export_dms(matches, "science-dms.csv")
    export_count("science-dms.csv")
