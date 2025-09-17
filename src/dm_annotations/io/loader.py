import gc
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from spacy.language import Language
from spacy.tokens import Doc, DocBin
from tqdm import tqdm

from dm_annotations.pipeline.cache import (
    batch,
    file_exists_and_complete,
    file_to_text_iter,
    get_cached_paths,
    text_to_sentence_iter,
)

# Centralised required metadata defaults
REQUIRED_META_DEFAULTS = {
    "title": "",
    "genre": "",
    "basename": "",
    "author": "",
    "year": None,
    "sentence_id": 0,
    "paragraph_id": None,
    "section": None,
}


def normalize_meta(meta: dict | None) -> dict:
    """Return a copy of REQUIRED_META_DEFAULTS with values from `meta` overriding defaults."""
    out = REQUIRED_META_DEFAULTS.copy()
    if isinstance(meta, dict):
        for k, default in REQUIRED_META_DEFAULTS.items():
            val = meta.get(k, default)
            if val is None:
                if k == "section":
                    out[k] = "unknown"
                else:
                    out[k] = default
            elif isinstance(default, str):
                out[k] = str(val or "")
            else:
                out[k] = val
    return out


def split_long_text(
    text: str, max_bytes: int = 49000, max_chars: int = 500
) -> list[str]:
    """
    Split text into chunks that don't exceed max_bytes when encoded as UTF-8.
    Tries to split on sentence boundaries first, then on whitespace, then force-splits.
    Discards sentences longer than max_chars that cannot be split on sentence boundaries.
    """
    if len(text.encode("utf-8")) <= max_bytes:
        return [text]

    chunks = []
    current_chunk = ""

    # Try splitting on sentence boundaries first
    sentences = text.split("。")

    for i, sentence in enumerate(sentences):
        # Add back the period except for the last sentence
        if i < len(sentences) - 1:
            sentence += "。"

        # Check if this single sentence is too long
        if len(sentence.encode("utf-8")) > max_bytes:
            # Try splitting on other sentence boundaries
            sub_sentences = []
            for delimiter in ["．", "！", "？", "\n"]:
                if delimiter in sentence:
                    sub_sentences = sentence.split(delimiter)
                    # Add back delimiters except for last part
                    for j in range(len(sub_sentences) - 1):
                        sub_sentences[j] += delimiter
                    break

            if sub_sentences and len(sub_sentences) > 1:
                # Process sub-sentences recursively
                for sub_sentence in sub_sentences:
                    if sub_sentence.strip():
                        sub_chunks = split_long_text(
                            sub_sentence.strip(), max_bytes, max_chars
                        )
                        chunks.extend(sub_chunks)
                continue

            # If sentence is still too long and exceeds character limit, discard it
            if len(sentence) > max_chars:
                import logging

                logging.warning(
                    f"Discarding sentence longer than {max_chars} characters: {sentence[:100]}..."
                )
                continue

            # Single sentence is too long, need to split it further
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""

            # Split the long sentence on whitespace
            words = sentence.split()
            for word in words:
                test_chunk = current_chunk + (" " if current_chunk else "") + word
                if len(test_chunk.encode("utf-8")) <= max_bytes:
                    current_chunk = test_chunk
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                        current_chunk = word

                    # If single word is still too long, force split by characters
                    while len(current_chunk.encode("utf-8")) > max_bytes:
                        # Binary search for the right split point
                        left, right = 0, len(current_chunk)
                        while left < right:
                            mid = (left + right + 1) // 2
                            if len(current_chunk[:mid].encode("utf-8")) <= max_bytes:
                                left = mid
                            else:
                                right = mid - 1

                        if left > 0:
                            chunks.append(current_chunk[:left])
                            current_chunk = current_chunk[left:]
                        else:
                            # Fallback: take first character to avoid infinite loop
                            chunks.append(current_chunk[:1])
                            current_chunk = current_chunk[1:]
        else:
            # Normal case: sentence fits within limits
            test_chunk = current_chunk + sentence
            if len(test_chunk.encode("utf-8")) <= max_bytes:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    # Filter out any remaining chunks that are too long
    filtered_chunks = []
    for chunk in chunks:
        if len(chunk.encode("utf-8")) <= max_bytes:
            filtered_chunks.append(chunk)
        else:
            import logging

            logging.warning(
                f"Discarding chunk that still exceeds byte limit: {chunk[:100]}..."
            )

    return filtered_chunks


def split_paragraphs(text: str) -> list[str]:
    """
    Split text into paragraphs on blank lines (two or more newlines), preserving
    intra-paragraph single newlines (one sentence per line).

    Examples:
    >>> split_paragraphs("A\\nB\\n\\nC\\n\\nD\\nE")
    ['A\\nB', 'C', 'D\\nE']
    >>> split_paragraphs("\\n\\nPara1 line1\\nPara1 line2\\n\\n\\nPara2\\n\\n")
    ['Para1 line1\\nPara1 line2', 'Para2']
    >>> split_paragraphs("")
    []
    """
    parts: list[str] = []
    cur: list[str] = []
    for ln in text.splitlines():
        if ln.strip():
            cur.append(ln)
        else:
            if cur:
                parts.append("\n".join(cur))
                cur.clear()
    if cur:
        parts.append("\n".join(cur))
    return parts


class CorpusParser:
    def __init__(
        self,
        source: Path | Iterator[tuple[str, dict[str, Any]]],
        nlp: Language,
        cache_batch_size: int = 100_000,
        pipe_batch_size: int = 1000,
        tokenize_only: bool = False,
        cache_only: bool = False,
        n_process: int | None = None,
    ):
        self.source = source
        self.nlp = nlp
        self.cache_batch_size = cache_batch_size
        self.pipe_batch_size = pipe_batch_size
        self.tokenize_only = tokenize_only
        self.cache_only = cache_only

        # Decide on number of processes for spaCy's nlp.pipe
        try:
            if n_process is None:
                n_procs = int(os.getenv("DM_NLP_PROCS", "1"))
            else:
                n_procs = int(n_process)
        except Exception:
            n_procs = 1

        self.n_process = max(1, n_procs)
        self._next_sentence_id: int = 0

    def _get_sentence_iterator(self) -> Iterator[tuple[str, dict[str, Any]]]:
        if isinstance(self.source, Path):
            with open(self.source, "rb") as f:
                yield from text_to_sentence_iter(file_to_text_iter(f))
        else:
            yield from self.source

    def stream(self) -> Iterator[Doc]:
        # For cache, if self.source is a Path, use as before.
        # If it's an iterator, use a synthetic cache path (not persistent).
        if isinstance(self.source, Path):
            cache_path, parsed_path, total_batches, batch_sizes = get_cached_paths(
                self.source, self.nlp, self.cache_batch_size
            )

            if any(
                not file_exists_and_complete(
                    cache_path / f"{parsed_path.stem}-{i}.spacy"
                )
                for i in range(total_batches)
            ):
                print("Cache missing or incomplete, regenerating...")
                disable_list = (
                    ["dm_extractor"] if "dm_extractor" in self.nlp.pipe_names else []
                )
                with self.nlp.select_pipes(disable=disable_list):
                    for _ in self._parse_or_get_docs():
                        pass

            extractor = (
                self.nlp.get_pipe("dm_extractor")
                if "dm_extractor" in self.nlp.pipe_names
                else None
            )
            for batch_num in tqdm(range(total_batches), desc="Processing batch"):
                batch_file = cache_path / f"{parsed_path.stem}-{batch_num}.spacy"
                for doc in tqdm(
                    DocBin(store_user_data=True)
                    .from_disk(batch_file)
                    .get_docs(self.nlp.vocab),
                    total=batch_sizes[batch_num],
                    desc="Doc",
                ):
                    # Ensure meta exists and all required fields are correct type
                    doc.user_data["meta"] = normalize_meta(doc.user_data.get("meta"))

                    # Always run dm_extractor right before yielding, if present
                    if extractor is not None:
                        doc = extractor(doc)
                    yield doc
        else:
            sentence_iter = self._get_sentence_iterator()

            def iter_chunks():
                for text, context in sentence_iter:
                    paras = split_paragraphs(text)
                    for p_idx, para in enumerate(paras):
                        ctx = dict(context)
                        if ctx.get("paragraph_id") is None:
                            ctx["paragraph_id"] = p_idx
                        # Do not discard long lines; only enforce byte cap
                        for chunk in split_long_text(
                            para, max_bytes=49000, max_chars=1_000_000
                        ):
                            yield (chunk, ctx)

            extractor = (
                self.nlp.get_pipe("dm_extractor")
                if "dm_extractor" in self.nlp.pipe_names
                else None
            )
            disable_list = (
                ["dm_extractor"] if "dm_extractor" in self.nlp.pipe_names else []
            )
            pipe_kwargs = {"as_tuples": True, "batch_size": self.pipe_batch_size}
            if getattr(self, "n_process", 1) > 1:
                pipe_kwargs["n_process"] = self.n_process

            with self.nlp.select_pipes(disable=disable_list):
                for doc, context in tqdm(self.nlp.pipe(iter_chunks(), **pipe_kwargs)):
                    # Attach normalized metadata before running extractor
                    genre = context.get("genre")
                    if isinstance(genre, list):
                        genre = genre[0] if genre else ""
                    meta_dict = {
                        "title": str(context.get("title") or ""),
                        "genre": str(genre or ""),
                        "basename": str(context.get("basename") or ""),
                        "author": str(context.get("author") or ""),
                        "year": context.get("year"),
                        "sentence_id": self._next_sentence_id,
                        "paragraph_id": context.get("paragraph_id"),
                        "section": context.get("section"),
                    }
                    doc.user_data["meta"] = normalize_meta(meta_dict)
                    self._next_sentence_id += 1

                    # Run extractor after meta is present (single call per doc)
                    if extractor is not None:
                        doc.user_data["meta"] = normalize_meta(
                            doc.user_data.get("meta")
                        )
                        doc = extractor(doc)
                        dm_list = []
                        for m in doc._.dm_matches:
                            dm_list.append(
                                {
                                    **{k: v for k, v in m.items() if k != "span"},
                                    "span": f"{m['span'].start_char}:{m['span'].end_char}",
                                }
                            )
                        doc.user_data["dm_matches"] = dm_list
                        doc._.dm_matches = []
                    yield doc

    def _parse_or_get_docs(self) -> Iterator[Doc]:
        cache_path, parsed_path, total_batches, batch_sizes = get_cached_paths(
            self.source, self.nlp, self.cache_batch_size
        )

        with open(self.source, "rb") as f:
            for batch_num, sentence_tuples_batch in enumerate(
                tqdm(
                    batch(
                        text_to_sentence_iter(file_to_text_iter(f)),
                        n=self.cache_batch_size,
                    ),
                    total=total_batches,
                    desc="_parse_or_get_docs",
                )
            ):
                batch_file = cache_path / f"{parsed_path.stem}-{batch_num}.spacy"
                if file_exists_and_complete(batch_file):
                    if self.cache_only:
                        continue
                    extractor = (
                        self.nlp.get_pipe("dm_extractor")
                        if "dm_extractor" in self.nlp.pipe_names
                        else None
                    )
                    for doc in tqdm(
                        DocBin(store_user_data=True)
                        .from_disk(batch_file)
                        .get_docs(self.nlp.vocab),
                        total=batch_sizes[batch_num],
                        desc="Docbin",
                    ):
                        # Ensure meta is present and all required fields are strings
                        doc.user_data["meta"] = normalize_meta(
                            doc.user_data.get("meta")
                        )
                        if extractor is not None:
                            doc = extractor(doc)
                        yield doc
                else:
                    if batch_file.exists():
                        print(f"Incomplete file {batch_file}, re-analyzing...")
                    doc_bin = DocBin(store_user_data=True)

                    with tqdm(total=batch_sizes[batch_num]) as pbar:
                        if self.tokenize_only:
                            for text, context in sentence_tuples_batch:
                                doc = self.nlp.make_doc(text)
                                pbar.update(1)
                                meta_dict = {
                                    "title": str(context.get("title") or ""),
                                    "genre": str(
                                        (
                                            context.get("genre")[0]
                                            if isinstance(context.get("genre"), list)
                                            and context.get("genre")
                                            else context.get("genre", "")
                                        )
                                        or ""
                                    ),
                                    "basename": str(context.get("basename") or ""),
                                    "author": str(context.get("author") or ""),
                                    "year": context.get("year"),
                                    "sentence_id": self._next_sentence_id,
                                    "paragraph_id": None,
                                    "section": None,
                                }
                                doc.user_data["meta"] = normalize_meta(meta_dict)
                                self._next_sentence_id += 1
                                yield doc
                                doc_bin.add(doc)
                        else:
                            # Build a generator of (chunk, context) so we do not materialize all chunks
                            def iter_chunks():
                                for text, context in sentence_tuples_batch:
                                    for chunk in split_long_text(text):
                                        yield (chunk, context)

                            disable_list = (
                                ["dm_extractor"]
                                if "dm_extractor" in self.nlp.pipe_names
                                else []
                            )
                            pipe_kwargs = {
                                "as_tuples": True,
                                "batch_size": self.pipe_batch_size,
                            }
                            if getattr(self, "n_process", 1) > 1:
                                pipe_kwargs["n_process"] = self.n_process

                            with self.nlp.select_pipes(disable=disable_list):
                                for doc, context in self.nlp.pipe(
                                    iter_chunks(), **pipe_kwargs
                                ):
                                    pbar.update(1)
                                    meta_dict = {
                                        "title": str(context.get("title") or ""),
                                        "genre": str(
                                            (
                                                context.get("genre")[0]
                                                if isinstance(
                                                    context.get("genre"), list
                                                )
                                                and context.get("genre")
                                                else context.get("genre", "")
                                            )
                                            or ""
                                        ),
                                        "basename": str(context.get("basename") or ""),
                                        "author": str(context.get("author") or ""),
                                        "year": context.get("year"),
                                        "sentence_id": self._next_sentence_id,
                                        "paragraph_id": None,
                                        "section": None,
                                    }
                                    doc.user_data["meta"] = normalize_meta(meta_dict)
                                    self._next_sentence_id += 1

                                    if not self.cache_only:
                                        yield doc
                                    doc_bin.add(doc)

                            gc.collect()
                            doc_bin.to_disk(
                                f"{cache_path}/{parsed_path.stem}-{batch_num}.spacy"
                            )
                            gc.collect()
