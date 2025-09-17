from collections.abc import Iterator
from io import BufferedReader
from itertools import islice
from pathlib import Path
from typing import TypeVar

import orjson
import spacy
import xxhash
from spacy.language import Language
from spacy.tokens import DocBin


def file_to_text_iter(f: BufferedReader) -> Iterator[dict]:
    for line in f:
        yield orjson.loads(line)


def text_to_sentence_iter(xs: Iterator[dict]) -> Iterator[tuple[str, dict[str, str]]]:
    for x in xs:
        for sentence in x["sentences"]:
            yield (sentence, {"genre": x["genre"], "title": x["title"]})


T = TypeVar("T")


def batch(xs: Iterator[T], n: int) -> Iterator[list[T]]:
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(xs)
    while batch := list(islice(it, n)):
        yield batch


def file_exists_and_complete(file_path: Path) -> bool:
    """Check if a file exists and is not 0-byte (complete)."""
    return file_path.exists() and file_path.stat().st_size > 0


def get_cached_paths(path: Path, nlp: Language, batch_size: int = 100000):
    hash_obj = xxhash.xxh64()
    # Hash values affecting outcome:
    hash_obj.update(spacy.__version__.encode())
    hash_obj.update(nlp.meta["name"].encode())
    hash_obj.update(nlp.meta["version"].encode())
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_obj.update(chunk)

    hash_string = hash_obj.hexdigest()

    # Get cache (base) path
    cache_path = Path("cache")
    cache_path.mkdir(exist_ok=True)
    # include batch_size in the cache‐directory name so you can easily distinguish runs
    parsed_path = cache_path / Path(
        path.stem
        + f"-{spacy.__version__}"
        + f"-{nlp.meta['name']}"
        + f"-{nlp.meta['version']}"
        + f"-batch{batch_size}"
        + f"-{hash_string}"
    )
    # file_pattern = f"{parsed_path.stem}-*.spacy"

    # Determine the total number of batches expected
    with open(path, "rb") as f:
        total_docs = sum(1 for _ in text_to_sentence_iter(file_to_text_iter(f)))
    total_batches = (total_docs + batch_size - 1) // batch_size
    batch_sizes = {
        batch: (batch_size if batch != total_batches - 1 else total_docs % batch_size)
        for batch in range(total_batches)
    }
    return cache_path, parsed_path, total_batches, batch_sizes


def parse_batch(batch_file, nlp_vocab):
    return list(DocBin().from_disk(batch_file).get_docs(nlp_vocab))
