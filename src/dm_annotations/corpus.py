from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
import gc
from io import BufferedReader
from itertools import islice
import os
from pathlib import Path

import orjson
import pandas as pd
import spacy
from spacy.language import Language
from spacy.tokens import Doc, DocBin
from tqdm import tqdm
import xxhash

from dm_annotations.matcher import (
    connectives_match,
    create_connectives_matcher,
    create_sf_matcher,
    pattern_match,
)
from dm_annotations.patterns import connectives_classifications, sf_classifications


def reserialize_corpus_jsonl(path: Path):
    title, genre = "", []
    sentences: list[str] = []
    with open(path, "rb") as f:
        for line in f:
            d = orjson.loads(line)
            if title and d["title"] != title:
                yield {
                    "title": title,
                    "genre": genre,
                    "sentences": sentences,
                }
                # TODO: These are also sentence-level: "tags": d["tags"],
                sentences = [d["text"]]
                title = d["title"]
                genre = d["genre"]
            else:
                title = d["title"]
                genre = d["genre"]
                sentences.append(d["text"])
    if sentences:
        yield {
            "title": title,
            "genre": genre,
            "sentences": sentences,
        }


def file_to_text_iter(f: BufferedReader) -> Iterator[dict]:
    for line in f:
        yield orjson.loads(line)


def text_to_sentence_iter(xs: Iterator[dict]) -> Iterator[tuple[str, dict[str, str]]]:
    for x in xs:
        for sentence in x["sentences"]:
            yield (sentence, {"genre": x["genre"], "title": x["title"]})


def batch(xs: Iterator[str], n: int) -> Iterator[list[str]]:
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
    parsed_path = cache_path / Path(
        path.stem
        + f"-{spacy.__version__}-{nlp.meta['name']}-{nlp.meta['version']}-{hash_string}"
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


def parse_docs(
    path: Path,
    nlp: Language,
    batch_size: int = 100000,
) -> Iterator[Doc]:
    cache_path, parsed_path, total_batches, batch_sizes = get_cached_paths(
        path, nlp, batch_size
    )

    for batch_num in tqdm(range(total_batches)):
        batch_file = cache_path / f"{parsed_path.stem}-{batch_num}.spacy"
        if file_exists_and_complete(batch_file):
            for doc in tqdm(
                DocBin().from_disk(batch_file).get_docs(nlp.vocab),
                total=batch_sizes[batch_num],
            ):
                yield doc
        else:
            raise Exception(f"Incomplete file {batch_file}, re-analyze.")


# from memory_profiler import profile
# @profile
def parse_or_get_docs(
    path: Path,
    nlp: Language,
    batch_size: int = 100000,
    tokenize_only: bool = False,
    cache_only: bool = False,
) -> Iterator[Doc]:
    cache_path, parsed_path, total_batches, batch_sizes = get_cached_paths(
        path, nlp, batch_size
    )

    # Use multiprocessing if not on GPU; clamp max processes to 8 for perf/memory sweet spot
    processes = min(os.cpu_count() or 4 // 2, 4) if not spacy.prefer_gpu() else 1
    print(f"Using {processes} processes for {path}.")

    # Process and yield each batch in order
    with open(path, "rb") as f:
        for batch_num, sentence_tuples_batch in enumerate(
            tqdm(
                batch(text_to_sentence_iter(file_to_text_iter(f)), n=batch_size),
                total=total_batches,
            )
        ):
            batch_file = cache_path / f"{parsed_path.stem}-{batch_num}.spacy"
            if file_exists_and_complete(batch_file):
                if cache_only:
                    continue
                for doc in tqdm(
                    DocBin().from_disk(batch_file).get_docs(nlp.vocab),
                    total=batch_sizes[batch_num],
                ):
                    yield doc
            else:
                if batch_file.exists():
                    print(f"Incomplete file {batch_file}, re-analyzing...")
                doc_bin = DocBin(store_user_data=True)

                with tqdm(total=batch_sizes[batch_num]) as pbar:
                    if tokenize_only:
                        # Note that with nlp.make_doc, only a straighforward tokenization is performed,
                        # and UD POS mappings are not fully aligned with the guidelines (AUX/NOUN -> VERB, etc.).
                        # Any analysis at this level should only use features available from Sudachi as-is.
                        for text, context in sentence_tuples_batch:
                            doc = nlp.make_doc(text)
                            pbar.update(1)
                            doc.user_data["genre"] = context["genre"]
                            doc.user_data["title"] = context["title"]
                            yield doc
                            doc_bin.add(doc)
                    else:
                        for doc, context in nlp.pipe(
                            sentence_tuples_batch,
                            as_tuples=True,
                            batch_size=5000,  # for sentences, a larger number (5000) is recommended
                            n_process=processes,
                        ):
                            pbar.update(1)
                            doc.user_data["genre"] = context["genre"]
                            doc.user_data["title"] = context["title"]
                            if not cache_only:
                                yield doc
                            doc_bin.add(doc)
                gc.collect()  # TODO Even with this, there seems to be a memory release issue.
                doc_bin.to_disk(f"{cache_path}/{parsed_path.stem}-{batch_num}.spacy")
                gc.collect()


def extract_dm(docs: Iterator[Doc], nlp: Language) -> list[dict]:
    nlp, c_matcher = create_connectives_matcher(nlp=nlp)
    nlp, sf_matcher = create_sf_matcher(nlp=nlp)
    for i, doc in enumerate(docs):
        c_matches = [
            {
                "span": span,
                "表現": span.label_,
                "タイプ": "接続表現",
                "機能": connectives_classifications[span.label_][
                    0
                ],  # Extract 1-5 from string start
                "細分類": connectives_classifications[span.label_],  # The whole string
                "position": span.start_char / len(span.doc.text),
                "ジャンル": doc.user_data["genre"][0],
                "title": doc.user_data["title"],
                "sentence_id": i,
            }
            for span in connectives_match(doc, nlp, c_matcher)
        ]
        sf_matches = [
            {
                "span": span,
                "表現": span.label_,
                "タイプ": "文末表現",
                "機能": sf_classifications[span.label_][0],
                "細分類": sf_classifications[span.label_][1],
                "position": span.start_char / len(span.doc.text),
                "ジャンル": doc.user_data["genre"][0],
                "title": doc.user_data["title"],
                "sentence_id": i,
            }
            for span in pattern_match(doc, nlp, sf_matcher)
        ]
        linear_matches = sorted(
            c_matches + sf_matches, key=lambda d: d["span"].start_char
        )
        yield linear_matches


def export_dms(matches, file_name: str):
    df = pd.DataFrame.from_records([pattern for match in matches for pattern in match])
    df.to_csv(file_name, index=False)
    try:
        df.to_excel(Path(file_name).stem + ".xlsx", index=False)
    except ValueError as e:  # If we exceed 1M rows, ignore and use CSV only
        print(f"Skipping Excel export: {e}")


def export_count(file_name: str):
    df = pd.read_csv(file_name)
    grouped = df.groupby(["タイプ", "ジャンル", "機能", "細分類", "表現"]).size()

    result_df = grouped.reset_index(name="頻度")
    result_df = result_df.sort_values(
        by=["タイプ", "ジャンル", "頻度"], ascending=[True, True, False]
    )
    result_df.to_csv(f"{Path(file_name).stem}-counts.csv", index=False)
    result_df.to_excel(f"{Path(file_name).stem}-counts.xlsx", index=False)


def run_thread_wrapped():
    return ThreadPoolExecutor().submit(foo).result()


if __name__ == "__main__":
    # Corpus preprocessing
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
    nlp = spacy.load(SPACY_MODEL, disable=["ner"])
    if "ginza" in SPACY_MODEL:
        nlp.add_pipe("disable_sentencizer", before="parser")

    # docs = parse_or_get_docs(Path("learner-corpus.jsonl"), nlp)
    # # Extract dms
    # matches = list(extract_dm(docs, nlp))
    # print(len(matches), len([match for match in matches if match]))
    # export_dms(matches, "learner-dms.csv")
    # export_count("learner-dms.csv")

    # docs = parse_or_get_docs(Path("science-corpus.jsonl"), nlp, cache_only=False)
    docs = parse_docs(Path("science-corpus.jsonl"), nlp)
    # Extract dms
    matches = list(extract_dm(docs, nlp))
    print(len(matches), len([match for match in matches if match]))
    export_dms(matches, "science-dms.csv")
    export_count("science-dms.csv")
