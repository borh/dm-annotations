import atexit
import html
import inspect
import logging
import random
import sqlite3
from functools import reduce
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable, Optional, Sequence

import altair as alt
import numpy as np
import orjson
import pandas as pd
import polars as pl
import prince
import spacy
import xxhash
from scipy.stats import chi2 as _chi2
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

# Import to register spaCy components
import dm_annotations.pipeline  # noqa: F401
from dm_annotations import load_dm_nlp
from dm_annotations.io.corpus import parse_docs
from dm_annotations.io.loader import CorpusParser
from dm_annotations.io.text_loader import (
    SECTION_EXCLUDE_DEFAULT,
    _get_md_parse_executor,
    _preprocess_and_write_markdown,
    _read_sources,
    parse_markdown_blocks_cached,
    parse_plain_folder_to_tuples,
)

logging.basicConfig(level=logging.INFO, force=True)


def make_segment_df_polars(
    sections: Iterable[dict[str, Any]],
    segment_chars: int = 60,
) -> pl.DataFrame:
    """
    Build a segmented Polars DataFrame from sections.
    Required keys per section: 'title', 'category', 'text'.
    """
    data = list(sections)
    base = pl.DataFrame(
        {
            "section_index": list(range(len(data))),
            "section_title": [
                s.get("title") or f"Section {i + 1}" for i, s in enumerate(data)
            ],
            "section_category": [s.get("category") or "Uncategorized" for s in data],
            "section_number": [s.get("section_number") for s in data],
            "text": [s.get("text") or "" for s in data],
        }
    )
    seg = int(segment_chars)
    df = (
        base.with_columns(pl.col("text").str.len_chars().alias("length"))
        .with_columns(
            pl.when(pl.col("length") > 0)
            .then((pl.col("length") + seg - 1) // seg)
            .otherwise(1)
            .alias("nseg")
        )
        .with_columns(pl.int_ranges(0, pl.col("nseg")).alias("segment_index_list"))
        .explode("segment_index_list")
        .rename({"segment_index_list": "segment_index"})
        .with_columns(
            [
                (pl.col("segment_index") * seg).alias("start_char"),
                pl.min_horizontal(
                    pl.col("length"), (pl.col("segment_index") + 1) * seg
                ).alias("end_char"),
            ]
        )
        .with_columns(
            pl.col("text")
            .str.slice(
                pl.col("start_char"),
                (pl.col("end_char") - pl.col("start_char")),
            )
            .alias("segment_text")
        )
        .with_row_index("global_index")
        .with_columns(
            pl.when(pl.col("segment_text").str.len_chars() > 50)
            .then(pl.col("segment_text").str.slice(0, 50) + pl.lit("..."))
            .otherwise(pl.col("segment_text"))
            .alias("segment_preview")
        )
    )
    return df


def add_wrap_columns(df_pl: pl.DataFrame, wrap_cols: int) -> pl.DataFrame:
    """Add 'col' and 'row' indices based on global_index and wrap width."""
    w = max(1, int(wrap_cols))
    return df_pl.with_columns(
        [
            (pl.col("global_index") % w).alias("col"),
            (pl.col("global_index") // w).alias("row"),
        ]
    )


def doc_mini_map_chart(df_pl: pl.DataFrame, wrap_cols: int = 20, tile_size: int = 6):
    """Static mini map chart using wrap_cols x tile_size."""
    df_wrapped = add_wrap_columns(df_pl, wrap_cols)
    n_rows = (
        1 + int(df_wrapped.select(pl.col("row").max()).to_series()[0])
        if df_wrapped.height
        else 1
    )
    # Limit dataset columns to reduce serialized size
    df_chart = df_wrapped.select(
        [
            "col",
            "row",
            "section_category",
            "section_title",
            "section_number",
            "segment_preview",
        ]
    )
    chart = (
        alt.Chart(df_chart)
        .mark_rect(stroke="#ffffff", strokeWidth=0.8)
        .encode(
            x=alt.X(
                "col:O", axis=None, scale=alt.Scale(paddingInner=0.2, paddingOuter=0.1)
            ),
            y=alt.Y(
                "row:O", axis=None, scale=alt.Scale(paddingInner=0.2, paddingOuter=0.1)
            ),
            color=alt.Color("section_category:N", legend=alt.Legend(title="Category")),
            tooltip=[
                alt.Tooltip("section_title:N", title="Section"),
                alt.Tooltip("section_number:N", title="Number"),
                alt.Tooltip("section_category:N", title="Category"),
                alt.Tooltip("segment_preview:N", title="Preview"),
            ],
        )
        .properties(
            width=int(wrap_cols) * int(tile_size), height=n_rows * int(tile_size)
        )
        .configure_view(stroke=None)
    )
    return chart


def doc_mini_map_chart_interactive(
    df_pl: pl.DataFrame, width: int = 800, height: int = 300, wrap_default: int = 20
):
    """Interactive mini map with a slider controlling wrap width."""
    wrap = alt.param(
        name="wrap",
        value=int(wrap_default),
        bind=alt.binding_range(name="Tiles per row", min=10, max=250, step=10),
    )
    # Limit dataset columns to reduce serialized size
    df_chart = df_pl.select(
        [
            "global_index",
            "section_category",
            "section_title",
            "section_number",
            "segment_preview",
        ]
    )
    chart = (
        alt.Chart(df_chart)
        .add_params(wrap)
        .transform_calculate(
            col="datum.global_index % wrap", row="floor(datum.global_index / wrap)"
        )
        .mark_rect(stroke="#ffffff", strokeWidth=0.8)
        .encode(
            x=alt.X(
                "col:O", axis=None, scale=alt.Scale(paddingInner=0.2, paddingOuter=0.1)
            ),
            y=alt.Y(
                "row:O", axis=None, scale=alt.Scale(paddingInner=0.2, paddingOuter=0.1)
            ),
            color=alt.Color("section_category:N", legend=alt.Legend(title="Category")),
            tooltip=[
                alt.Tooltip("section_title:N", title="Section"),
                alt.Tooltip("section_number:N", title="Number"),
                alt.Tooltip("section_category:N", title="Category"),
                alt.Tooltip("segment_preview:N", title="Preview"),
            ],
        )
        .properties(width=int(width), height=int(height))
        .configure_view(stroke=None)
    )
    return chart


def build_minimap_df_from_nodes(
    nodes: list[Any], segment_chars: int = 200
) -> pl.DataFrame:
    """
    Convert SectionNode list (with .raw_text, .category, .body_text) into a segmented DF.
    """
    try:
        from dm_annotations.io.text_loader import _pick_primary_category
    except Exception:

        def _pick_primary_category(c):  # type: ignore
            if not c:
                return None
            if isinstance(c, (set, list, tuple)):
                return sorted(list(c))[0]
            return str(c)

    sections: list[dict[str, Any]] = []
    for i, n in enumerate(nodes):
        body = getattr(n, "body_text", "") or ""
        if not body:
            continue
        cat = _pick_primary_category(getattr(n, "category", None))
        title = getattr(n, "raw_text", None) or (cat or f"Section {i + 1}")
        sections.append(
            {
                "title": str(title),
                "category": str(cat) if cat else "Uncategorized",
                "text": body,
                "section_number": (str(getattr(n, "num_prefix", "")) or None),
            }
        )
    return make_segment_df_polars(sections, segment_chars=segment_chars)


def build_doc_minimap_for_file(
    file_path: Path,
    *,
    segment_chars: int = 200,
    wrap_cols: int = 20,
    tile_size: int = 6,
    interactive: bool = False,
    skip_classification: bool | None = None,
    section_exclude: set[str] | None = None,
):
    """
    Parse a markdown/text file into sections and return a mini map Altair chart.
    """
    from dm_annotations.io.text_loader import (
        SECTION_EXCLUDE_DEFAULT,
        parse_md_to_section_nodes,
    )

    sec_excl = (
        section_exclude if section_exclude is not None else set(SECTION_EXCLUDE_DEFAULT)
    )
    nodes = parse_md_to_section_nodes(
        file_path, section_exclude=sec_excl, skip_classification=skip_classification
    )
    df_pl = build_minimap_df_from_nodes(nodes, segment_chars=segment_chars)
    if interactive:
        return doc_mini_map_chart_interactive(df_pl, wrap_default=wrap_cols)
    return doc_mini_map_chart(df_pl, wrap_cols=wrap_cols, tile_size=tile_size)


def _xxh(b: bytes) -> str:
    return xxhash.xxh64(b).hexdigest()


def _hash_file(path: Path) -> str:
    try:
        with open(path, "rb") as f:
            return _xxh(f.read())
    except Exception:
        return "NA"


def _fingerprint_patterns() -> str:
    """
    Return a short digest for invalidation when patterns/model change.
    Hashes dm_annotations.pipeline.patterns.py and any resources/*patterns*.json.
    """
    try:
        import dm_annotations.pipeline.patterns as pmod

        p_path = Path(getattr(pmod, "__file__", ""))
        parts: list[tuple[str, str]] = []
        if p_path and p_path.exists():
            parts.append(("patterns_py", _hash_file(p_path)))
            root = p_path.parent.parent.parent
            res_dir = root / "resources"
            if res_dir.exists():
                for rp in sorted(res_dir.glob("*patterns*.json")):
                    parts.append((rp.name, _hash_file(rp)))
        return _xxh(orjson.dumps(parts, option=orjson.OPT_SORT_KEYS))
    except Exception:
        return "NA"


class SentenceCache:
    """
    SQLite-backed cache storing per-sentence DM matches keyed by (namespace, key).
    key = xxhash64(sentence bytes). value = orjson-encoded list of matches with
    fields: タイプ, 表現, 機能, position, end_position (positions are sentence-relative).
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _init(self) -> None:
        con = sqlite3.connect(self.db_path)
        try:
            con.execute("PRAGMA journal_mode=WAL;")
            con.execute("PRAGMA synchronous=NORMAL;")
            con.execute(
                """CREATE TABLE IF NOT EXISTS sentence_cache (
                ns TEXT NOT NULL, key TEXT NOT NULL, text_len INTEGER NOT NULL,
                value BLOB NOT NULL, PRIMARY KEY(ns, key))"""
            )
            con.commit()
        finally:
            con.close()

    def get_many(self, ns: str, keys: list[tuple[str, int]]) -> dict[str, bytes]:
        if not keys:
            return {}
        con = sqlite3.connect(self.db_path)
        try:
            out: dict[str, bytes] = {}
            # Chunk IN clause to avoid SQLite parameter limits
            for i in range(0, len(keys), 1000):
                chunk = keys[i : i + 1000]
                ks = [k for (k, _) in chunk]
                rows = con.execute(
                    f"SELECT key, text_len, value FROM sentence_cache "
                    f"WHERE ns=? AND key IN ({','.join(['?'] * len(ks))})",
                    [ns, *ks],
                ).fetchall()
                for k, tlen, val in rows:
                    out[k] = val
            return out
        finally:
            con.close()

    def put_many(self, ns: str, rows: list[tuple[str, int, bytes]]) -> None:
        if not rows:
            return
        con = sqlite3.connect(self.db_path)
        try:
            con.executemany(
                "INSERT OR REPLACE INTO sentence_cache(ns,key,text_len,value) VALUES(?,?,?,?)",
                [(ns, k, tlen, val) for (k, tlen, val) in rows],
            )
            con.commit()
        finally:
            con.close()


def _dedup_anchors_payload(payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, Any, Any, Any]] = set()
    out: list[dict[str, Any]] = []
    for m in payload:
        key = (
            m.get("タイプ"),
            m.get("表現"),
            m.get("position"),
            m.get("end_position"),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(m)
    return out


def _prepare_df_entries(
    df_input: pl.DataFrame,
    basename_filter: set[str] | None,
    remove_basenames: set[str] | None,
    section_exclude: set[str] | None,
    sample: float,
) -> list[dict[str, Any]]:
    """Filter, sample, and flatten a DataFrame of sections into unique sentence entries."""
    # Pre-emptive check for required columns in the input
    for col in ["basename", "corpus", "genre1", "genre2"]:
        if col not in df_input.columns:
            raise ValueError(
                f"Input DataFrame to _prepare_df_entries is missing required column: {col}"
            )
        if df_input.filter(pl.col(col).is_null()).height > 0:
            logging.warning(
                f"Input DataFrame to _prepare_df_entries has null values in column '{col}'."
            )

    df_work = df_input
    if basename_filter is not None:
        df_work = df_work.filter(pl.col("basename").is_in(list(basename_filter)))
    if remove_basenames:
        df_work = df_work.filter(~pl.col("basename").is_in(list(remove_basenames)))
    if section_exclude:
        df_work = df_work.filter(~pl.col("section").is_in(list(section_exclude)))
    if sample < 1.0:
        df_work = df_work.sample(fraction=sample, seed=42)

    entries: list[dict[str, Any]] = []
    for row in df_work.iter_rows(named=True):
        section_text = row.get("text") or ""
        meta_base = {k: v for k, v in row.items() if k != "text"}

        for para_text in section_text.split("\n\n"):
            para_text = para_text.strip()
            if not para_text:
                continue

            meta = meta_base.copy()
            meta["paragraph_text"] = para_text

            for line in para_text.splitlines():
                if not line.strip():
                    continue
                bn = str(meta.get("basename") or "")
                sec = str(meta.get("section") or "")
                sid_key = f"{bn}|{sec}|{line}"
                sid = (
                    xxhash.xxh64(sid_key.encode("utf-8")).intdigest()
                    & 0x7FFFFFFFFFFFFFFF
                )
                entries.append(
                    {
                        "key": _xxh(line.encode("utf-8")),
                        "text": line,
                        "tlen": len(line),
                        "meta": meta,
                        "sid": sid,
                    }
                )

    orig_n = len(entries)
    sid_seen: set[int] = set()
    dedup_entries: list[dict[str, Any]] = []
    for e in entries:
        sid = e["sid"]
        if sid in sid_seen:
            continue
        sid_seen.add(sid)
        dedup_entries.append(e)
    logging.info(
        "sentence-cache: unique sentences=%d (from=%d)", len(dedup_entries), orig_n
    )
    return dedup_entries


def _process_df_cache_hits(
    entries: list[dict[str, Any]], scache: SentenceCache, ns: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Process cache hits, repair cache, and return hit records and misses."""
    if not entries:
        return [], []

    cached_hits = scache.get_many(ns, [(e["key"], e["tlen"]) for e in entries])
    logging.info("sentence-cache: entries=%d, hits=%d", len(entries), len(cached_hits))

    records: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    to_update: list[tuple[str, int, bytes]] = []

    for e in entries:
        key, line, tlen, meta, sid = (
            e["key"],
            e["text"],
            e["tlen"],
            e["meta"],
            e["sid"],
        )
        raw = cached_hits.get(key)
        if raw is not None:
            try:
                payload = orjson.loads(raw)
            except Exception:
                missing.append(e)
                continue

            payload2 = _dedup_anchors_payload(payload)
            if len(payload2) != len(payload):
                to_update.append((key, tlen, orjson.dumps(payload2)))

            if payload2:
                for m in payload2:
                    rec = {**meta, **m, "sentence_id": sid, "sentence_text": line}
                    rec["span_text"] = (
                        line[m["position"] : m["end_position"]]
                        if m.get("position") is not None
                        and m.get("end_position") is not None
                        else ""
                    )
                    records.append(rec)
            continue
        missing.append(e)

    if to_update:
        try:
            scache.put_many(ns, to_update)
            logging.info(
                "sentence-cache: repaired %d cached sentences (removed duplicate anchors)",
                len(to_update),
            )
        except Exception:
            logging.debug("Failed to repair sentence-cache entries", exc_info=True)

    return records, missing


def _process_df_cache_misses(
    missing: list[dict[str, Any]],
    nlp: Any,
    scache: SentenceCache,
    ns: str,
    cache_batch_size: int,
    pipe_batch_size: int,
    n_process: int,
) -> list[dict[str, Any]]:
    """Process cache misses through spaCy and return new records."""
    if not missing:
        return []

    logging.info("sentence-cache: misses=%d", len(missing))

    def _iter_missing():
        for e in missing:
            yield (e["text"], e["meta"])

    parser = CorpusParser(
        _iter_missing(),
        nlp,
        cache_batch_size=cache_batch_size,
        pipe_batch_size=pipe_batch_size,
        n_process=n_process,
    )
    docs = parser.stream()
    records: list[dict[str, Any]] = []
    to_store: list[tuple[str, int, bytes]] = []

    for doc in docs:
        line = doc.text
        key = _xxh(line.encode("utf-8"))
        tlen = len(line)
        meta = getattr(doc, "user_data", {}).get("meta", {}) or {}
        bn = str(meta.get("basename") or "")
        sec = str(meta.get("section") or "")
        sid = (
            xxhash.xxh64(f"{bn}|{sec}|{line}".encode("utf-8")).intdigest()
            & 0x7FFFFFFFFFFFFFFF
        )

        doc_user_data = getattr(doc, "user_data", None)
        ud_dm = (
            doc_user_data.get("dm_matches") if isinstance(doc_user_data, dict) else None
        )
        ext_dm = getattr(getattr(doc, "_", None), "dm_matches", None)
        dm_matches = ud_dm or ext_dm or []

        payload: list[dict[str, Any]] = []
        seen_anchors: set[tuple[Any, Any, Any, Any]] = set()

        for m in dm_matches:
            sv = m.get("span")
            s_i, e_i = None, None
            if isinstance(sv, str):
                try:
                    s_s, e_s = sv.split(":", 1)
                    s_i, e_i = int(s_s), int(e_s)
                except Exception:
                    pass
            else:
                s_i, e_i = (
                    getattr(sv, "start_char", None),
                    getattr(sv, "end_char", None),
                )

            key_anchor = (m.get("タイプ"), m.get("表現"), s_i, e_i)
            if key_anchor in seen_anchors:
                continue
            seen_anchors.add(key_anchor)

            match_data = {
                "タイプ": m.get("タイプ"),
                "表現": m.get("表現"),
                "機能": m.get("機能"),
                "position": s_i,
                "end_position": e_i,
            }
            payload.append(match_data)
            records.append(
                {
                    **meta,
                    **match_data,
                    "sentence_id": sid,
                    "sentence_text": line,
                    "span_text": line[s_i:e_i]
                    if s_i is not None and e_i is not None
                    else "",
                }
            )

        if scache and ns:
            to_store.append((key, tlen, orjson.dumps(payload)))
            if len(to_store) >= 2000:
                scache.put_many(ns, to_store)
                to_store.clear()

    if scache and ns and to_store:
        scache.put_many(ns, to_store)

    return records


def _extract_from_df_source(
    src: pl.DataFrame | pd.DataFrame,
    model: str,
    use_sentence_cache: bool,
    sentence_cache_path: Path | str,
    cache_version: str,
    **kwargs: Any,
) -> pl.DataFrame:
    """Orchestrates DM extraction from a DataFrame source."""
    df_input = src if isinstance(src, pl.DataFrame) else pl.from_pandas(src)
    logging.info("Processing in-memory DataFrame source with %d rows", len(df_input))

    if not use_sentence_cache:
        raise NotImplementedError(
            "Non-cached DataFrame extraction path is not implemented in this refactor."
        )

    ns = f"{cache_version}|{model}|{spacy.__version__}|{_fingerprint_patterns()}"
    scache = SentenceCache(Path(sentence_cache_path))

    prepare_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k in {"basename_filter", "remove_basenames", "section_exclude", "sample"}
    }
    entries = _prepare_df_entries(df_input, **prepare_kwargs)
    hit_records, missing = _process_df_cache_hits(entries, scache, ns)

    miss_records = []
    if missing:
        nlp = load_dm_nlp(model, disable=["ner"])
        process_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in {"cache_batch_size", "pipe_batch_size", "n_process"}
        }
        miss_records = _process_df_cache_misses(
            missing, nlp, scache, ns, **process_kwargs
        )

    all_records = hit_records + miss_records
    return pl.DataFrame(all_records) if all_records else pl.DataFrame()


def _get_docs_from_path(
    src: Path,
    model: str,
    sample: float,
    basename_filter: set[str] | None,
    remove_basenames: set[str] | None,
    section_exclude: set[str] | None,
    **kwargs: Any,
) -> tuple[Any, Iterable]:
    """Handles file sampling and gets a Doc iterator for a Path source."""
    nlp = load_dm_nlp(model, disable=["ner"])
    sentence_tuples: Iterable[tuple[str, dict[str, Any]]]

    if src.is_dir():
        file_filter = basename_filter
        if sample < 1.0:
            sources_tsv = src / "sources.tsv"
            if sources_tsv.exists():
                sources_df = _read_sources(sources_tsv)
                sample_size = max(1, int(len(sources_df) * sample))
                sampled_basenames = set(
                    random.sample(sources_df["basename"].to_list(), sample_size)
                )
            else:
                all_files = list(src.glob("*.md")) or list(src.glob("*.txt"))
                sample_size = max(1, int(len(all_files) * sample))
                sampled_basenames = set(
                    random.sample([f.stem for f in all_files], sample_size)
                )
            file_filter = (
                (file_filter & sampled_basenames) if file_filter else sampled_basenames
            )

        if remove_basenames:
            if file_filter is None:
                all_files = list(src.glob("*.md")) or list(src.glob("*.txt"))
                file_filter = {p.stem for p in all_files}
            file_filter -= remove_basenames

        sentence_tuples = parse_plain_folder_to_tuples(
            src,
            strict_metadata=kwargs.get("strict_metadata", True),
            file_filter=file_filter,
            section_exclude=section_exclude,
        )
        parser = CorpusParser(sentence_tuples, nlp, **kwargs)
        return nlp, parser.stream()
    else:  # JSONL file
        if sample < 1.0:
            with NamedTemporaryFile(
                mode="wb", suffix=".jsonl", delete=False
            ) as temp_file:
                with open(src, "rb") as f:
                    for line in f:
                        if random.random() < sample:
                            temp_file.write(line)
                temp_jsonl_path = Path(temp_file.name)
            atexit.register(lambda: temp_jsonl_path.unlink(missing_ok=True))
            return nlp, parse_docs(temp_jsonl_path, nlp, **kwargs)
        else:
            return nlp, parse_docs(src, nlp, **kwargs)


def _process_docs_to_records(docs: Iterable) -> Iterable[dict[str, Any]]:
    """Generator that processes an iterator of spaCy Docs and yields DM match records."""
    sentence_global_id = 0
    for doc in docs:
        line_ranges: list[tuple[int, int, int]] = []
        pos = 0
        for part in doc.text.splitlines(True):
            raw = part.rstrip("\r\n")
            start = pos
            end = start + len(raw)
            pos += len(part)
            if raw.strip():
                line_ranges.append((start, end, sentence_global_id))
                sentence_global_id += 1
        line_by_id = {sid: (s, e) for s, e, sid in line_ranges}

        def _sid_for_start(start_char: int | None) -> int | None:
            if start_char is None:
                return None
            for s, e, sid in line_ranges:
                if s <= start_char < e:
                    return sid
            return None

        doc_user_data = getattr(doc, "user_data", {})
        dm_matches = doc_user_data.get("dm_matches") or getattr(
            getattr(doc, "_", None), "dm_matches", []
        )
        if not dm_matches:
            continue

        seen_anchors: set[tuple[Any, Any, Any, Any]] = set()
        for m in dm_matches:
            rec = {k: v for k, v in m.items() if k != "span"}
            span_val = m.get("span")
            start_char, end_char = None, None
            if isinstance(span_val, str):
                try:
                    start_s, end_s = span_val.split(":", 1)
                    start_char, end_char = int(start_s), int(end_s)
                    rec["span_text"] = doc.text[start_char:end_char]
                except Exception:
                    rec["span_text"] = ""
            else:
                start_char = getattr(span_val, "start_char", None)
                end_char = getattr(span_val, "end_char", None)
                rec["span_text"] = getattr(span_val, "text", "")

            rec["position"], rec["end_position"] = start_char, end_char
            key_anchor = (rec.get("タイプ"), rec.get("表現"), start_char, end_char)
            if key_anchor in seen_anchors:
                continue
            seen_anchors.add(key_anchor)

            sid = _sid_for_start(start_char)
            if sid is not None and sid in line_by_id:
                s_start, s_end = line_by_id[sid]
                rec["sentence_text"] = doc.text[s_start:s_end]
            else:
                rec["sentence_text"] = None
            rec["sentence_id"] = sid if sid is not None else -1
            rec.update(doc_user_data.get("meta", {}))
            yield rec


def _extract_from_path_source(
    src: Path, model: str, section_exclude: set[str] | None, **kwargs: Any
) -> pl.DataFrame:
    """Orchestrates DM extraction from a Path source (directory or file)."""
    _, docs = _get_docs_from_path(src, model, section_exclude=section_exclude, **kwargs)
    records = list(_process_docs_to_records(docs))

    if section_exclude:
        exclude = set(section_exclude)
        before = len(records)
        records = [
            r
            for r in records
            if not (
                r.get("section")
                and (
                    r["section"] in exclude
                    if isinstance(r["section"], str)
                    else any(s in exclude for s in r["section"])
                )
            )
        ]
        logging.info(
            "Removed %d records from excluded sections: %s",
            before - len(records),
            sorted(exclude),
        )
    return pl.DataFrame(records) if records else pl.DataFrame()


def _validate_output_df(out: pl.DataFrame) -> None:
    """Runs a series of diagnostic assertions on the final DataFrame."""
    if out.is_empty():
        return

    for col in ["corpus", "genre1", "genre2"]:
        if col not in out.columns:
            raise AssertionError(f"Output DataFrame is missing required column: {col}")

        null_rows = out.filter(pl.col(col).is_null())
        if not null_rows.is_empty():
            sample = null_rows.head(5).select(
                ["basename", "title", "section", "sentence_text"]
            )
            raise AssertionError(
                f"Output DataFrame column '{col}' contains null values. "
                f"Found {len(null_rows)} nulls. Sample of failing rows: {sample.to_dicts()}"
            )

    try:
        # 1) Check for identical anchor duplicates (low-level bug check)
        anchor_dups = (
            out.group_by(["sentence_id", "タイプ", "表現", "position", "end_position"])
            .len()
            .filter(pl.col("len") > 1)
        )
        if anchor_dups.height > 0:
            logging.error(
                "Duplicate identical anchors (sentence_id,タイプ,表現,position,end_position) detected: total=%d; sample=%s",
                anchor_dups.height,
                anchor_dups.head(5).to_dicts(),
            )

        # 2) Binary reduction for validation
        dfu = (
            out.select(["sentence_id", "タイプ", "表現", "position"])
            .drop_nulls(["sentence_id", "タイプ", "表現"])
            .group_by(["sentence_id", "タイプ", "表現"])
            .agg(pl.col("position").min().alias("pos"))
        )
        conn_df = (
            dfu.filter(pl.col("タイプ") == "接続表現")
            .select(["sentence_id", "表現"])
            .rename({"表現": "接続表現"})
            .unique()
        )

        # 3) Assert: at most one unique 接続表現 per sentence
        multi_conn_sents = (
            conn_df.group_by("sentence_id").len().filter(pl.col("len") > 1)
        )
        if multi_conn_sents.height > 0:
            sample_sids = multi_conn_sents.head(5).get_column("sentence_id").to_list()
            sample_details = (
                out.filter(
                    pl.col("sentence_id").is_in(sample_sids)
                    & (pl.col("タイプ") == "接続表現")
                )
                .select(["sentence_id", "表現", "position", "sentence_text"])
                .unique()
                .sort("sentence_id", "position")
            )
            raise AssertionError(
                f"Found {multi_conn_sents.height} sentences with more than one unique 接続表現. "
                f"This violates the assumption of one connective per sentence. "
                f"Sample details: {sample_details.to_dicts()}"
            )

        end_df = (
            dfu.filter(pl.col("タイプ") == "文末表現")
            .select(["sentence_id", "表現"])
            .rename({"表現": "文末表現"})
            .unique()
        )

        # 4) Assert: no duplicate (conn, end) pairs per sentence after binary reduction
        if not conn_df.is_empty() and not end_df.is_empty():
            pairs = conn_df.join(end_df, on="sentence_id", how="inner")
            dup_pairs = (
                pairs.group_by(["sentence_id", "接続表現", "文末表現"])
                .len()
                .filter(pl.col("len") > 1)
            )
            if dup_pairs.height > 0:
                details = []
                for dp in dup_pairs.head(5).to_dicts():
                    sid = dp["sentence_id"]
                    cx, ex = dp["接続表現"], dp["文末表現"]
                    sid_rows = out.filter(pl.col("sentence_id") == sid)

                    def _first(col: str):
                        return (
                            sid_rows.select(pl.col(col))
                            .drop_nulls()
                            .limit(1)
                            .to_series()
                            .item(0, None)
                            if col in sid_rows.columns
                            else None
                        )

                    details.append(
                        {
                            "sentence_id": sid,
                            "接続表現": cx,
                            "文末表現": ex,
                            "meta": {
                                "basename": _first("basename"),
                                "section": _first("section"),
                                "title": _first("title"),
                                "year": _first("year"),
                                "sentence_text": _first("sentence_text"),
                            },
                        }
                    )
                raise AssertionError(
                    "Duplicate 接続表現×文末表現 pairs after binary reduction: "
                    f"{dup_pairs.height} duplicates. Sample details: "
                    + orjson.dumps(details, option=orjson.OPT_INDENT_2).decode("utf-8")
                )
    except AssertionError:
        raise
    except Exception:
        pass


def extract_dms_df(
    src: Path | pl.DataFrame | pd.DataFrame,
    model: str = "ja_ginza",
    cache_batch_size: int = 100_000,
    pipe_batch_size: int = 1000,
    n_process: int = 1,
    strict_metadata: bool = True,
    sample: float = 1.0,
    section_exclude: set[str] | None = None,
    basename_filter: set[str] | None = None,
    remove_basenames: set[str] | None = None,
    use_sentence_cache: bool = True,
    sentence_cache_path: Path | str = "cache/dm_sentence_cache.sqlite",
    cache_version: str = "sent-v1",
) -> pl.DataFrame:
    """
    Extract DM matches from a source and return as a Polars DataFrame.

    This function acts as a dispatcher, handling either a DataFrame source (with sentence-level caching)
    or a Path source (directory or JSONL file). It ensures data integrity by running a series of
    diagnostic assertions on the final output.

    :param src: Path to a directory/JSONL file or a Polars/Pandas DataFrame of sections.
    :param model: spaCy model name.
    :param cache_batch_size: Sentences per cache file (for CorpusParser).
    :param pipe_batch_size: spaCy .pipe batch size.
    :param n_process: Number of processes for spaCy nlp.pipe.
    :param strict_metadata: If True, require valid metadata (title, year, genre).
    :param sample: Proportion of data to sample (0.0-1.0).
    :param section_exclude: Set of normalized section names to exclude.
    :param basename_filter: Set of basenames to include (for directory sources).
    :param remove_basenames: Set of basenames to exclude.
    :param use_sentence_cache: If True, use the sentence-level cache for DataFrame sources.
    :param sentence_cache_path: Path to the SQLite sentence cache database.
    :param cache_version: Version string for the sentence cache namespace.
    :return: A Polars DataFrame containing the extracted DM matches.
    """
    is_df_source = isinstance(src, (pl.DataFrame, pd.DataFrame))
    kwargs = {
        "cache_batch_size": cache_batch_size,
        "pipe_batch_size": pipe_batch_size,
        "n_process": n_process,
        "strict_metadata": strict_metadata,
        "sample": sample,
        "section_exclude": section_exclude,
        "basename_filter": basename_filter,
        "remove_basenames": remove_basenames,
    }

    if is_df_source:
        out = _extract_from_df_source(
            src,
            model,
            use_sentence_cache,
            sentence_cache_path,
            cache_version,
            **kwargs,
        )
    else:
        src_path = Path(src).expanduser()
        out = _extract_from_path_source(src_path, model, **kwargs)

    _validate_output_df(out)
    return out


def count_dms_df(
    df: pl.DataFrame,
    group_cols: Sequence[str] = ("タイプ", "ジャンル", "機能", "細分類", "表現"),
) -> pl.DataFrame:
    """
    Count DM matches by one or more columns.

    :param df: output of extract_dms_df
    :param group_cols: columns to group by
    :return: grouped & sorted DataFrame with '頻度'
    """
    gb = df.group_by(list(group_cols)).len(name="頻度")
    # sort by all group_cols then descending freq
    return gb.sort(group_cols + ["頻度"], descending=[False] * len(group_cols) + [True])


def filter_by(
    df: pl.DataFrame,
    expr: str | None = None,
    types: Sequence[str] | None = None,
    genres: Sequence[str] | None = None,
) -> pl.DataFrame:
    """
    Filter DataFrame by expression substring, DM types, or genres.

    :param df: DataFrame from extract_dms_df
    :param expr: substring or regex to match in 'span_text'
    :param types: list of 'タイプ' values to include
    :param genres: list of 'ジャンル' values to include
    """
    q = df
    if expr:
        q = q.filter(pl.col("span_text").str.contains(expr))
    if types:
        q = q.filter(pl.col("タイプ").is_in(list(types)))
    if genres:
        q = q.filter(pl.col("ジャンル").is_in(list(genres)))
    return q


def compute_pmi_entropy(
    df: pl.DataFrame,
    category_col: str = "タイプ",
    group_cols: Sequence[str] | None = None,
) -> pl.DataFrame:
    """
    Compute pointwise mutual information (PMI), entropy, and transition probabilities
    for collocations that appear in the same sentence in the given DM matches DataFrame.

    Behavior:
    - The DM category column is by default 'タイプ' (no autodetection). If you
      pass another column name explicitly it will be used instead.
    - If group_cols is provided (e.g. ["ジャンル"] or ["ジャンル","section"])
      PMI/entropy are computed independently for each distinct group combination
      and the group columns are included in the returned table.
    - Transition probabilities are calculated based on sequential position within sentences.
    - Pair counting is binary: per sentence, each ordered pair (expr1, expr2) is counted at most once.

    :param df: DataFrame from extract_dms_df() with at least 'sentence_id' and '表現' columns (position auto-extracted from spans)
    :param category_col: column name containing a category label for each expression (default: 'タイプ')
    :param group_cols: optional sequence of column names to group by before computing PMI/entropy
    :return: Polars DataFrame with columns:
             (group_cols...,) ['接続表現','文末表現','joint_count','count_接続表現','count_文末表現','p_xy','p_x','p_y','pmi','entropy_接続表現','entropy_文末表現','transition_conn_to_end']

    Examples:
    >>> import polars as pl
    >>> small = pl.DataFrame([
    ...     {"sentence_id": 1, "表現": "a1", "タイプ": "接続表現"},
    ...     {"sentence_id": 1, "表現": "a2", "タイプ": "接続表現"},
    ...     {"sentence_id": 1, "表現": "b1", "タイプ": "文末表現"},
    ...     {"sentence_id": 1, "表現": "b2", "タイプ": "文末表現"},
    ... ])
    >>> res = compute_pmi_entropy(small)  # uses default category_col='タイプ'
    >>> sorted([(r["接続表現"], r["文末表現"], int(r["joint_count"])) for r in res.select(["接続表現","文末表現","joint_count"]).to_dicts()])  # doctest: +NORMALIZE_WHITESPACE
    [('a1', 'b1', 1), ('a1', 'b2', 1), ('a2', 'b1', 1), ('a2', 'b2', 1)]

    >>> small2 = pl.DataFrame([
    ...     {"sentence_id": 1, "表現": "a1", "タイプ": "接続表現", "ジャンル": "g1"},
    ...     {"sentence_id": 1, "表現": "b1", "タイプ": "文末表現", "ジャンル": "g1"},
    ...     {"sentence_id": 2, "表現": "a2", "タイプ": "接続表現", "ジャンル": "g2"},
    ...     {"sentence_id": 2, "表現": "b2", "タイプ": "文末表現", "ジャンル": "g2"},
    ... ])
    >>> res2 = compute_pmi_entropy(small2, group_cols=["ジャンル"])
    >>> sorted([(r["ジャンル"], r["接続表現"], r["文末表現"], int(r["joint_count"])) for r in res2.select(["ジャンル","接続表現","文末表現","joint_count"]).to_dicts()])  # doctest: +NORMALIZE_WHITESPACE
    [('g1', 'a1', 'b1', 1), ('g2', 'a2', 'b2', 1)]
    """
    required_cols = {"sentence_id", "表現", "position"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    if category_col is None:
        raise ValueError("category_col must be provided (default is 'タイプ')")

    if category_col not in df.columns:
        raise ValueError(
            f"category_col '{category_col}' not found in DataFrame columns"
        )

    if group_cols:
        for c in group_cols:
            if c not in df.columns:
                raise ValueError(f"group column '{c}' not found in DataFrame columns")

    def _compute_for_df(local_df: pl.DataFrame) -> pl.DataFrame:
        # Count unique sentence occurrences per expression (for marginals and total sentences)
        expr_sentence = local_df.select(["sentence_id", "表現"]).unique()
        total_sentences = int(expr_sentence["sentence_id"].n_unique())

        # Build one row per (sentence_id, 表現, category) keeping the earliest position
        dfu = (
            local_df.select(["sentence_id", "表現", category_col, "position"])
            .drop_nulls(["sentence_id", "表現"])
            .group_by(["sentence_id", "表現", category_col])
            .agg(pl.col("position").min().alias("pos"))
        )

        # Split by type and assign canonical column names and positions
        conn_df = (
            dfu.filter(pl.col(category_col) == "接続表現")
            .rename({"表現": "接続表現", "pos": "conn_pos"})
            .select(["sentence_id", "接続表現", "conn_pos"])
        )
        end_df = (
            dfu.filter(pl.col(category_col) == "文末表現")
            .rename({"表現": "文末表現", "pos": "end_pos"})
            .select(["sentence_id", "文末表現", "end_pos"])
        )

        # Build cross-type pairs within each sentence
        pairs_df = conn_df.join(end_df, on="sentence_id", how="inner")
        pairs_df = pairs_df.unique(subset=["sentence_id", "接続表現", "文末表現"])

        if pairs_df.is_empty():
            return pl.DataFrame([])

        # Joint counts (unordered across direction but fixed type-order 接続表現×文末表現)
        joint_counts = pairs_df.group_by(["接続表現", "文末表現"]).agg(
            pl.len().alias("joint_count")
        )

        # Marginal counts per expression across sentences
        count_conn = (
            conn_df.select(["sentence_id", "接続表現"])
            .unique()
            .group_by("接続表現")
            .agg(pl.len().alias("count_接続表現"))
        )
        count_end = (
            end_df.select(["sentence_id", "文末表現"])
            .unique()
            .group_by("文末表現")
            .agg(pl.len().alias("count_文末表現"))
        )

        # Attach marginals and probabilities
        joint_counts = (
            joint_counts.join(count_conn, on="接続表現", how="left")
            .join(count_end, on="文末表現", how="left")
            .with_columns(
                [
                    (pl.col("joint_count") / total_sentences).alias("p_xy"),
                    (pl.col("count_接続表現") / total_sentences).alias("p_x"),
                    (pl.col("count_文末表現") / total_sentences).alias("p_y"),
                ]
            )
            .with_columns(
                pl.when(
                    (pl.col("p_x") > 0) & (pl.col("p_y") > 0) & (pl.col("p_xy") > 0)
                )
                .then((pl.col("p_xy") / (pl.col("p_x") * pl.col("p_y"))).log(base=2))
                .otherwise(0.0)
                .alias("pmi")
            )
        )

        # Directed transition counts: 接続表現 precedes 文末表現 within the sentence
        dir_counts = (
            pairs_df.with_columns(
                (pl.col("conn_pos") < pl.col("end_pos")).alias("conn_before_end")
            )
            .filter(pl.col("conn_before_end"))
            .group_by(["接続表現", "文末表現"])
            .agg(pl.len().alias("conn_to_end_count"))
        )

        # Transition probability P(文末表現 | 接続表現) = conn_to_end_count / count_接続表現
        joint_counts = (
            joint_counts.join(dir_counts, on=["接続表現", "文末表現"], how="left")
            .with_columns(pl.col("conn_to_end_count").fill_null(0).cast(pl.Int64))
            .with_columns(
                pl.when(pl.col("count_接続表現") > 0)
                .then(pl.col("conn_to_end_count") / pl.col("count_接続表現"))
                .otherwise(0.0)
                .alias("transition_conn_to_end")
            )
            .drop(["conn_to_end_count"])
        )

        # Entropy over neighbor distributions (vectorized with Polars)
        # For each 接続表現, distribution over 文末表現
        conn_tot = joint_counts.group_by("接続表現").agg(
            pl.col("joint_count").sum().alias("_tot_conn")
        )
        conn_ent = (
            joint_counts.join(conn_tot, on="接続表現", how="left")
            .with_columns(
                (pl.col("joint_count") / pl.col("_tot_conn")).alias("_p_conn")
            )
            .with_columns(
                pl.when(pl.col("_p_conn") > 0)
                .then(-(pl.col("_p_conn") * pl.col("_p_conn").log(base=2)))
                .otherwise(0.0)
                .alias("_h_conn")
            )
            .group_by("接続表現")
            .agg(pl.col("_h_conn").sum().alias("entropy_接続表現"))
        )

        # For each 文末表現, distribution over 接続表現
        end_tot = joint_counts.group_by("文末表現").agg(
            pl.col("joint_count").sum().alias("_tot_end")
        )
        end_ent = (
            joint_counts.join(end_tot, on="文末表現", how="left")
            .with_columns((pl.col("joint_count") / pl.col("_tot_end")).alias("_p_end"))
            .with_columns(
                pl.when(pl.col("_p_end") > 0)
                .then(-(pl.col("_p_end") * pl.col("_p_end").log(base=2)))
                .otherwise(0.0)
                .alias("_h_end")
            )
            .group_by("文末表現")
            .agg(pl.col("_h_end").sum().alias("entropy_文末表現"))
        )

        joint_counts = joint_counts.join(conn_ent, on="接続表現", how="left").join(
            end_ent, on="文末表現", how="left"
        )

        return joint_counts

    # If grouping requested, run computation per-group and append group columns
    if group_cols:
        out_frames: list[pl.DataFrame] = []
        # iterate unique group combinations
        group_keys_df = df.select(list(group_cols)).unique()
        for group_row in group_keys_df.iter_rows(named=True):
            sub_df = df
            for k, v in group_row.items():
                if v is None:
                    sub_df = sub_df.filter(pl.col(k).is_null())
                else:
                    sub_df = sub_df.filter(pl.col(k) == v)
            sub_res = _compute_for_df(sub_df)
            if sub_res.is_empty():
                continue
            # add the group columns as constants
            for k, v in group_row.items():
                sub_res = sub_res.with_columns(pl.lit(v).alias(k))
            out_frames.append(sub_res)

        if not out_frames:
            return pl.DataFrame([])

        result = pl.concat(out_frames, how="vertical")
        # prefer group columns first in the returned schema
        ordered_cols = list(group_cols) + [
            c for c in result.columns if c not in group_cols
        ]
        return result.select(ordered_cols)

    # No grouping: compute on entire df
    return _compute_for_df(df)


def build_contingency_table(
    df: pl.DataFrame,
    category_col: str | Sequence[str] = "section",
    expr_col: str = "表現",
) -> pl.DataFrame:
    """
    Build a contingency table (category x expression counts) and return it as a Polars DataFrame.

    category_col may be a single column name (str) or a sequence of column names;
    when multiple columns are provided they are combined into a single composite index
    by joining values with " / ". The returned table has that index column first and
    one column per unique expression (values from `expr_col`) containing integer counts.

    Example:
    >>> import polars as pl
    >>> small = pl.DataFrame([
    ...     {"section": "g1", "表現": "a"},
    ...     {"section": "g1", "表現": "b"},
    ...     {"section": "g2", "表現": "a"},
    ... ])
    >>> build_contingency_table(small, category_col="section", expr_col="表現").to_dicts()  # doctest: +NORMALIZE_WHITESPACE
    [{'section': 'g1', 'a': 1, 'b': 1}, {'section': 'g2', 'a': 1, 'b': 0}]
    """
    if df.is_empty():
        return pl.DataFrame([])

    # Normalize category_col to a list of column names
    if isinstance(category_col, (list, tuple)):
        cat_cols = list(category_col)
    else:
        cat_cols = [category_col]

    # Validate requested columns exist
    missing = [c for c in cat_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Category column(s) not found in DataFrame: {missing}")

    # If multiple category columns were requested, create a composite index column
    if len(cat_cols) == 1:
        index_col = cat_cols[0]
        ct_src = df
    else:
        index_col = " / ".join(cat_cols)
        try:
            ct_src = df.with_columns(
                pl.concat_str(
                    [pl.col(c).cast(pl.Utf8) for c in cat_cols], sep=" / "
                ).alias(index_col)
            )
        except Exception:
            # Fallback: convert to pandas and build composite column then back to Polars
            df_pd = df.to_pandas()
            df_pd[index_col] = df_pd[cat_cols].astype(str).agg(" / ".join, axis=1)
            ct_src = pl.from_pandas(df_pd)

    ct_pl = (
        ct_src.group_by([index_col, expr_col])
        .len()
        .pivot(index=index_col, on=expr_col, values="len", aggregate_function="sum")
        .fill_null(0)
    )

    # Ensure counts are integer-typed
    for col_name in ct_pl.columns:
        if col_name == index_col:
            continue
        ct_pl = ct_pl.with_columns(pl.col(col_name).cast(pl.Int64))

    return ct_pl


def chi2_standardized_residuals(
    df: pl.DataFrame,
    category_col: str | Sequence[str] = "section",
    expr_col: str = "表現",
    drop_zeros: bool = True,
) -> tuple[pl.DataFrame, float, int, float, float]:
    """
    Compute Chi-squared test of independence and standardized residuals for a
    category-by-expression contingency table, using Polars only for data ops.

    Returns a tidy residuals table and summary stats (chi2, dof, p-value, cramer's v).

    Args:
        df: Input matches with at least the category and expression columns.
        category_col: Category column (e.g., "section" or "ジャンル") or a sequence
            of columns to combine (same convention as build_contingency_table).
        expr_col: Expression column name, typically "表現".
        drop_zeros: When True (default), drop categories (rows) and expressions (columns)
            whose total counts are zero before computing expected counts and degrees of freedom.

    Returns:
        residuals: Polars DataFrame with columns:
            - category (index column name resolved from category_col)
            - expr_col (e.g., "表現")
            - observed (int)
            - expected (float)
            - std_resid (float; (O - E) / sqrt(E))
        chi2: Chi-square statistic (float)
        dof: Degrees of freedom (int)
        p_value: p-value (float)
        cramers_v: Cramer's V effect size (float)

    Examples:
    >>> import polars as pl
    >>> small = pl.DataFrame([
    ...     {"section":"A","表現":"x"},
    ...     {"section":"A","表現":"x"},
    ...     {"section":"A","表現":"y"},
    ...     {"section":"B","表現":"x"},
    ...     {"section":"B","表現":"y"},
    ...     {"section":"B","表現":"y"},
    ... ])
    >>> resid, chi2, dof, p, v = chi2_standardized_residuals(small, category_col="section", expr_col="表現")
    >>> dof
    1
    """
    # Build contingency (may combine multiple category columns into a composite index)
    ct_pl = build_contingency_table(df, category_col=category_col, expr_col=expr_col)
    if isinstance(category_col, (list, tuple)):
        index_col = " / ".join(category_col)
    else:
        index_col = category_col

    # Guard trivial cases
    if ct_pl.is_empty():
        return (
            pl.DataFrame(
                schema={
                    index_col: pl.Utf8,
                    expr_col: pl.Utf8,
                    "observed": pl.Int64,
                    "expected": pl.Float64,
                    "std_resid": pl.Float64,
                }
            ),
            0.0,
            0,
            1.0,
            0.0,
        )

    expr_cols = [c for c in ct_pl.columns if c != index_col]

    # Optionally drop empty rows/columns to avoid inflating dof/expectations
    if drop_zeros:
        # Drop rows with zero sum
        if expr_cols:
            ct_pl = (
                ct_pl.with_columns(
                    pl.sum_horizontal([pl.col(c) for c in expr_cols]).alias("_row_sum")
                )
                .filter(pl.col("_row_sum") > 0)
                .drop("_row_sum")
            )
            expr_cols = [c for c in ct_pl.columns if c != index_col]
        # Drop columns with zero sum
        if expr_cols:
            col_sums = ct_pl.select([pl.col(c).sum().alias(c) for c in expr_cols]).row(
                0, named=True
            )
            keep_cols = [c for c in expr_cols if int(col_sums.get(c, 0)) > 0]
            drop_cols = [c for c in expr_cols if c not in keep_cols]
            if drop_cols:
                ct_pl = ct_pl.drop(drop_cols)
            expr_cols = keep_cols

    if not expr_cols or ct_pl.height < 2 or len(expr_cols) < 2:
        return (
            pl.DataFrame(
                schema={
                    index_col: pl.Utf8,
                    expr_col: pl.Utf8,
                    "observed": pl.Int64,
                    "expected": pl.Float64,
                    "std_resid": pl.Float64,
                }
            ),
            0.0,
            0,
            1.0,
            0.0,
        )

    # Row sums and total N
    ct_rows = ct_pl.with_columns(
        pl.sum_horizontal([pl.col(c) for c in expr_cols]).alias("_row_sum")
    )
    N = int(ct_rows.select(pl.col("_row_sum").sum()).to_series()[0])
    if N <= 0:
        return (
            pl.DataFrame(
                schema={
                    index_col: pl.Utf8,
                    expr_col: pl.Utf8,
                    "observed": pl.Int64,
                    "expected": pl.Float64,
                    "std_resid": pl.Float64,
                }
            ),
            0.0,
            0,
            1.0,
            0.0,
        )

    # Column sums as scalars
    col_sums_row = ct_pl.select([pl.col(c).sum().alias(c) for c in expr_cols])
    col_sums = col_sums_row.row(0, named=True)

    # Expected counts per cell
    expected_defs = [
        (pl.col("_row_sum") * pl.lit(int(col_sums[c])) / pl.lit(N)).alias(f"_E_{c}")
        for c in expr_cols
    ]
    ct_E = ct_rows.with_columns(expected_defs)

    # Standardized residuals and chi-square contributions
    eps = 1e-12
    resid_defs = [
        (
            (pl.col(c) - pl.col(f"_E_{c}")) / (pl.col(f"_E_{c}") + pl.lit(eps)).sqrt()
        ).alias(f"_R_{c}")
        for c in expr_cols
    ]
    contrib_defs = [
        (
            (pl.col(c) - pl.col(f"_E_{c}")) ** 2 / (pl.col(f"_E_{c}") + pl.lit(eps))
        ).alias(f"_C_{c}")
        for c in expr_cols
    ]
    ct_RC = ct_E.with_columns(resid_defs + contrib_defs)

    # Chi-square statistic and p-value
    chi2_row = ct_RC.with_columns(
        pl.sum_horizontal([pl.col(f"_C_{c}") for c in expr_cols]).alias("_chi2_row")
    )
    chi2_stat = float(chi2_row.select(pl.col("_chi2_row").sum()).to_series()[0])
    dof = max(0, (ct_pl.height - 1) * (len(expr_cols) - 1))
    p_value = float(_chi2.sf(chi2_stat, dof)) if dof > 0 else 1.0

    # Cramer's V
    if N > 0 and dof > 0:
        k = len(expr_cols)
        r = ct_pl.height
        phi2 = chi2_stat / N
        min_dim = min(k, r) - 1
        cramers_v = np.sqrt(phi2 / min_dim) if min_dim > 0 else 0.0
    else:
        cramers_v = 0.0

    # Tidy long output: observed, expected, standardized residuals
    obs_long = ct_pl.unpivot(
        index=[index_col],
        variable_name=expr_col,
        value_name="observed",
    )
    exp_long = ct_E.select(
        [pl.col(index_col)] + [pl.col(f"_E_{c}").alias(c) for c in expr_cols]
    ).unpivot(index=[index_col], variable_name=expr_col, value_name="expected")
    res_long = ct_RC.select(
        [pl.col(index_col)] + [pl.col(f"_R_{c}").alias(c) for c in expr_cols]
    ).unpivot(index=[index_col], variable_name=expr_col, value_name="std_resid")

    residuals = obs_long.join(exp_long, on=[index_col, expr_col], how="inner").join(
        res_long, on=[index_col, expr_col], how="inner"
    )
    return residuals, chi2_stat, dof, p_value, cramers_v


def llr_keyness(
    df: pl.DataFrame,
    target: str,
    category_col: str | Sequence[str] = "section",
    expr_col: str = "表現",
    min_count: int = 5,
    correction: str | None = "fdr_bh",
) -> pl.DataFrame:
    """
    Compute keyness using the log-likelihood ratio (Rayson & Garside, LLR)
    comparing a target category vs. the rest, per expression.

    LLR is computed on 2x2 tables for each expression:
        target counts: a
        rest counts:   b
    with expectations:
        E1 = (a+b) * (N1/N)
        E2 = (a+b) * (N2/N)
    where N1 is total tokens in target, N2 in rest, N = N1 + N2.

    Returns a Polars DataFrame with:
        - expr_col (e.g., "表現")
        - llr (float, higher = more characteristic)
        - p_value (float, from chi2 distribution)
        - p_adj (float, corrected for multiple comparisons)
        - target_rel (float, a/N1 - b/N2; sign indicates direction)
        - a (int, count in target)
        - b (int, count in rest)

    Args:
        df: Input matches with at least category_col and expr_col.
        target: Category value to compare against the rest.
        category_col: Category column or sequence of columns to combine.
        expr_col: Expression column.
        min_count: Minimum (a+b) to keep an expression in the result.
        correction: Method for multiple comparison correction, passed to statsmodels.
                    Default is 'fdr_bh' (Benjamini-Hochberg). None to disable.

    Examples:
    >>> import polars as pl
    >>> small = pl.DataFrame([
    ...     {"section":"A","表現":"x"}, {"section":"A","表現":"x"}, {"section":"A","表現":"y"},
    ...     {"section":"B","表現":"x"}, {"section":"B","表現":"y"}, {"section":"B","表現":"y"},
    ... ])
    >>> out = llr_keyness(small, target="A", category_col="section", expr_col="表現", min_count=1)
    >>> set(out.columns) >= {"表現","llr","p_value","p_adj","target_rel","a","b"}
    True
    """
    # Build contingency; resolve index column name like build_contingency_table
    ct_pl = build_contingency_table(df, category_col=category_col, expr_col=expr_col)
    if isinstance(category_col, (list, tuple)):
        index_col = " / ".join(category_col)
    else:
        index_col = category_col

    if ct_pl.is_empty() or index_col not in ct_pl.columns:
        return pl.DataFrame([])

    # Long table of counts
    long = ct_pl.unpivot(index=[index_col], variable_name=expr_col, value_name="count")

    # Totals per expression and totals per category
    expr_totals = long.group_by(expr_col).agg(pl.col("count").sum().alias("n_total"))
    target_counts = (
        long.filter(pl.col(index_col) == target)
        .group_by(expr_col)
        .agg(pl.col("count").sum().alias("a"))
    )

    # Merge to get a (fill null with 0) and compute b = n_total - a
    counts = (
        expr_totals.join(target_counts, on=expr_col, how="left")
        .with_columns(pl.col("a").fill_null(0).cast(pl.Float64))
        .with_columns(pl.col("n_total").cast(pl.Float64))
        .with_columns((pl.col("n_total") - pl.col("a")).alias("b"))
    )

    # Totals N1 (target), N2 (rest), N overall
    N1 = int(
        long.filter(pl.col(index_col) == target)
        .select(pl.col("count").sum())
        .to_series()[0]
        or 0
    )
    N = int(expr_totals.select(pl.col("n_total").sum()).to_series()[0] or 0)
    N2 = max(0, N - N1)
    if N <= 0:
        return pl.DataFrame([])

    # Expectations per expression
    eps = 1e-12
    counts = counts.with_columns(
        [
            (pl.col("n_total") * (pl.lit(N1) / pl.lit(N))).alias("_E1"),
            (pl.col("n_total") * (pl.lit(N2) / pl.lit(N))).alias("_E2"),
        ]
    )

    # LLR terms with safe guards for zero counts
    term1 = (
        pl.when((pl.col("a") > 0) & (pl.col("_E1") > 0))
        .then(pl.col("a") * (pl.col("a") / (pl.col("_E1") + eps)).log())
        .otherwise(0.0)
    )
    term2 = (
        pl.when((pl.col("b") > 0) & (pl.col("_E2") > 0))
        .then(pl.col("b") * (pl.col("b") / (pl.col("_E2") + eps)).log())
        .otherwise(0.0)
    )

    out = counts.with_columns(
        [
            (2.0 * (term1 + term2)).alias("llr"),
            (
                (pl.col("a") / pl.lit(N1 if N1 > 0 else 1))
                - (pl.col("b") / pl.lit(N2 if N2 > 0 else 1))
            ).alias("target_rel"),
        ]
    ).with_columns([pl.col("a").cast(pl.Int64), pl.col("b").cast(pl.Int64)])

    # Filter by minimum total frequency
    out = out.filter((pl.col("a") + pl.col("b")) >= min_count)

    if out.is_empty():
        return pl.DataFrame([])

    # P-value from LLR (chi-squared distribution with 1 dof)
    out = out.with_columns(
        pl.col("llr").map_batches(lambda s: _chi2.sf(s.to_numpy(), 1)).alias("p_value")
    )

    # Correction for multiple comparisons
    if correction and not out.is_empty():
        p_values = out["p_value"].to_numpy()
        reject, p_adj, _, _ = multipletests(p_values, alpha=0.05, method=correction)
        out = out.with_columns(pl.Series("p_adj", p_adj))
        out = out.select(
            [pl.col(expr_col), "llr", "p_value", "p_adj", "target_rel", "a", "b"]
        ).sort(["llr"], descending=True)
    else:
        out = out.select(
            [pl.col(expr_col), "llr", "p_value", "target_rel", "a", "b"]
        ).sort(["llr"], descending=True)

    return out


def expand_function_columns(df: pl.DataFrame, col: str = "機能") -> pl.DataFrame:
    """
    Parse a '機能' column into type-specific structured columns.

    For 接続表現: splits "2整理・列挙・まず" into cumulative levels:
      - 機能_1: "2整理" (code + first field)
      - 機能_2: "2整理・列挙" (code + first + second, cumulative)
      - 機能_3: "2整理・列挙・まず" (all fields, cumulative)

    For 文末表現: different format with max 2 levels:
      - 機能_1: first part
      - 機能_2: first + second part (cumulative)

    Returns a Polars DataFrame with type-specific columns. The function is idempotent.
    """
    if df is None or df.is_empty() or "機能_1" in df.columns:
        return df

    # Normalize full-width digits
    full_width_digits = [chr(ord("０") + i) for i in range(10)]
    half_width_digits = [chr(ord("0") + i) for i in range(10)]
    norm_func_expr = reduce(
        lambda expr, p: expr.str.replace(p[0], p[1], literal=True, n=-1),
        zip(full_width_digits, half_width_digits),
        pl.col(col),
    )

    # --- Expressions for 接続表現 ---
    code_conn_expr = norm_func_expr.str.extract(r"^\s*([0-9]+)", 1).fill_null("")
    rest_conn_expr = norm_func_expr.str.replace(
        r"^\s*[0-9]+\s*", "", n=1
    ).str.strip_chars()
    tags_conn_expr = rest_conn_expr.str.extract_all(r"[^・、,/／;；|\s+]+")

    f1_conn_expr = pl.when(tags_conn_expr.list.len() > 0).then(
        code_conn_expr + tags_conn_expr.list.get(0)
    )
    f2_conn_expr = (
        pl.when(tags_conn_expr.list.len() >= 2)
        .then(code_conn_expr + tags_conn_expr.list.slice(0, 2).list.join("・"))
        .otherwise(f1_conn_expr)
    )
    f3_conn_expr = (
        pl.when(tags_conn_expr.list.len() >= 3)
        .then(code_conn_expr + tags_conn_expr.list.join("・"))
        .otherwise(f2_conn_expr)
    )

    # --- Expressions for 文末表現 ---
    # This logic is simplified to be backward-compatible and also fixes a bug
    # where leading delimiters caused incorrect parsing of the first part.
    DELIMITER_CHARS = "・、,/／;；| "
    stripped_expr = norm_func_expr.str.strip_chars(characters=DELIMITER_CHARS)

    # f1 is the first "word".
    f1_end_expr = stripped_expr.str.extract(f"^([^{DELIMITER_CHARS}]+)", 1)
    # f2 is the whole thing (after stripping delimiters from ends).
    f2_end_expr = stripped_expr

    if "タイプ" not in df.columns:
        # Fallback: treat all as 接続表現 style
        return df.with_columns(
            機能_1=f1_conn_expr,
            機能_2=f2_conn_expr,
            機能_3=f3_conn_expr,
        )

    # --- Combine based on タイプ ---
    conn_mask = pl.col("タイプ") == "接続表現"

    return df.with_columns(
        機能_1=pl.when(conn_mask).then(f1_conn_expr).otherwise(f1_end_expr),
        機能_2=pl.when(conn_mask).then(f2_conn_expr).otherwise(f2_end_expr),
        機能_3=pl.when(conn_mask).then(f3_conn_expr).otherwise(None),
    )


def visualize_ca(
    df: pl.DataFrame,
    category_col: str | Sequence[str] = "section",
    expr_col: str = "表現",
    random_state: int = 42,
    min_freq: int = 1,
):
    """
    Run Correspondence Analysis (CA) using Prince on DM data grouped by category.

    :param df: Polars DataFrame from extract_dms_df
    :param category_col: column to use as category ('ジャンル' or 'section', etc.)
    :param expr_col: column to use for expressions (default: '表現')
    :param random_state: random seed for CA reproducibility
    :param min_freq: Minimum frequency per item to display on the plot.
    :return: Matplotlib figure object from Prince plot() or DataFrame if 1D
    """

    # Build contingency table (use helper so it can be inspected separately in notebooks)
    ct_pl = build_contingency_table(df, category_col=category_col, expr_col=expr_col)

    # If caller passed multiple category columns, build_contingency_table created a
    # composite index named by joining those columns with " / ". Use that name here.
    if isinstance(category_col, (list, tuple)):
        index_col_used = " / ".join(category_col)
    else:
        index_col_used = category_col
    # Safety: check shape
    num_rows = ct_pl.shape[0]
    num_cols = ct_pl.shape[1] - 1  # minus index column
    max_comps = min(num_rows - 1, num_cols - 1)
    if num_rows < 2 or num_cols < 2:
        raise ValueError(
            f"Not enough categories/expressions for 2D CA plot: "
            f"{num_rows} categories × {num_cols} expressions (need ≥2 in each)."
        )

    # Convert to pandas for Prince
    ct = ct_pl.to_pandas().set_index(index_col_used)

    # Adjust n_components to data
    if max_comps < 2:
        logging.warning(
            f"Only {max_comps} CA component(s) available; falling back to 1D CA. "
            "Consider adding more category levels for 2D plots."
        )
        n_components = max_comps
    else:
        n_components = 2

    ca_model = prince.CA(
        n_components=n_components,
        n_iter=10,
        copy=True,
        check_input=True,
        engine="sklearn",
        random_state=random_state,
    ).fit(ct)

    if n_components < 2:
        # No 2D scatter possible; return row coordinates DataFrame for 1D
        coords = ca_model.row_coordinates(ct)
        print("1D CA result (row coordinates):")
        print(coords)
        return coords
    else:
        # Build an enhanced Altair chart from the CA coordinates so we can
        # tweak fonts, shapes and color-by-タイプ without re-implementing CA.
        try:
            eig = ca_model._eigenvalues_summary.to_dict(orient="index")
            total_inertia = ca_model.total_inertia_

            # Get coordinates (pandas DataFrames)
            row_coords = ca_model.row_coordinates(ct)
            col_coords = ca_model.column_coordinates(ct)
            # Rename coordinate columns like prince does
            row_coords.columns = [f"component {i}" for i in row_coords.columns]
            col_coords.columns = [f"component {i}" for i in col_coords.columns]

            # Add helper columns similar to prince.plot
            row_coords = row_coords.assign(
                variable=index_col_used, value=row_coords.index.astype(str)
            )
            col_coords = col_coords.assign(
                variable="expression", value=col_coords.index.astype(str)
            )

            # Map expression -> タイプ using the input DataFrame (df).
            # Also compute per-category and per-expression frequencies from the contingency table
            # so we can scale marker areas by frequency. Use pandas for convenience; if mapping fails, default to "unknown".
            type_col = "タイプ"
            expr_col_local = expr_col
            try:
                # Normalize incoming df to a pandas DataFrame
                if isinstance(df, pl.DataFrame):
                    df_pd = df.to_pandas()
                elif isinstance(df, pd.DataFrame):
                    df_pd = df
                else:
                    df_pd = pd.DataFrame(df)

                expr_type_map: dict[str, str] = {}
                expr_category_map: dict[str, str] = {}

                # Compute frequencies from the contingency table `ct` (pandas DataFrame)
                # row_freq_series: index -> count per category, col_freq_series: expression -> count
                try:
                    row_freq_series = ct.sum(axis=1)
                    col_freq_series = ct.sum(axis=0)
                except Exception:
                    # if ct is not available for any reason, fall back to zeros
                    row_freq_series = pd.Series(dtype="int64")
                    col_freq_series = pd.Series(dtype="int64")

                if expr_col_local in df_pd.columns:
                    # most frequent タイプ per 表現 (mode) if available
                    if type_col in df_pd.columns:
                        expr_type = df_pd.groupby(expr_col_local)[type_col].agg(
                            lambda s: s.value_counts().index[0]
                        )
                        expr_type_map = expr_type.to_dict()

                    # dominant category per 表現: support either a single column or a sequence
                    if isinstance(category_col, (list, tuple)):
                        missing = [c for c in category_col if c not in df_pd.columns]
                        if not missing:
                            combined_name = " / ".join(category_col)
                            # create a combined column in pandas for grouping
                            df_pd[combined_name] = (
                                df_pd[list(category_col)]
                                .astype(str)
                                .agg(" / ".join, axis=1)
                            )
                            expr_cat = df_pd.groupby(expr_col_local)[combined_name].agg(
                                lambda s: s.value_counts().index[0]
                            )
                            expr_category_map = expr_cat.to_dict()
                    else:
                        if category_col in df_pd.columns:
                            expr_cat = df_pd.groupby(expr_col_local)[category_col].agg(
                                lambda s: s.value_counts().index[0]
                            )
                            expr_category_map = expr_cat.to_dict()

                    col_coords["タイプ"] = [
                        expr_type_map.get(e, "unknown")
                        for e in col_coords.index.astype(str)
                    ]
                    col_coords["dominant_category"] = [
                        expr_category_map.get(e, "unknown")
                        for e in col_coords.index.astype(str)
                    ]
                else:
                    n = len(col_coords.index)
                    col_coords["タイプ"] = ["unknown"] * n
                    col_coords["dominant_category"] = ["unknown"] * n

                # Attach frequency counts to the coordinate frames for sizing encodings.
                # row_coords index corresponds to categories (index_col_used), col_coords index to expressions.
                try:
                    row_coords["freq"] = [
                        int(row_freq_series.get(idx, 0))
                        for idx in row_coords.index.astype(str)
                    ]
                except Exception:
                    row_coords["freq"] = [0] * len(row_coords.index)
                try:
                    col_coords["freq"] = [
                        int(col_freq_series.get(idx, 0))
                        for idx in col_coords.index.astype(str)
                    ]
                except Exception:
                    col_coords["freq"] = [0] * len(col_coords.index)

                # Apply min_freq filter to column coordinates before plotting.
                # Keep the CA model computed on the full contingency table but only plot
                # expressions/collocations that meet the requested display frequency.
                try:
                    if min_freq and int(min_freq) > 1:
                        col_coords = col_coords[col_coords["freq"] >= int(min_freq)]
                except Exception:
                    # Be conservative on failure: do not filter if anything goes wrong.
                    pass
            except Exception:
                n = len(col_coords.index)
                col_coords["タイプ"] = ["unknown"] * n
                col_coords["dominant_category"] = ["unknown"] * n
                row_coords["freq"] = [0] * len(row_coords.index)
                col_coords["freq"] = [0] * len(col_coords.index)

            # Prepare plotting DataFrames (ensure label column exists)
            row_plot_df = row_coords.reset_index(drop=True)
            row_plot_df["label"] = row_plot_df["value"].astype(str)
            col_plot_df = col_coords.reset_index(drop=True)
            col_plot_df["label"] = col_plot_df["value"].astype(str)

            # Remove rows with NaN or infinite coordinate values to avoid JSON serialization errors
            comp_x = "component 0"
            comp_y = "component 1"
            # Filter out rows where either plotted coordinate is NaN or ±Inf
            row_plot_df = row_plot_df[
                row_plot_df[comp_x].notna()
                & row_plot_df[comp_y].notna()
                & np.isfinite(row_plot_df[comp_x])
                & np.isfinite(row_plot_df[comp_y])
            ].reset_index(drop=True)
            col_plot_df = col_plot_df[
                col_plot_df[comp_x].notna()
                & col_plot_df[comp_y].notna()
                & np.isfinite(col_plot_df[comp_x])
                & np.isfinite(col_plot_df[comp_y])
            ].reset_index(drop=True)

            # Axis titles with explained variance (match prince formatting)
            x_label = f"component {0} — {eig[0]['% of variance'] / 100:.2%}"
            y_label = f"component {1} — {eig[1]['% of variance'] / 100:.2%}"

            base_enc = dict(
                x=alt.X(
                    "component 0",
                    scale=alt.Scale(zero=False),
                    axis=alt.Axis(title=x_label),
                ),
                y=alt.Y(
                    "component 1",
                    scale=alt.Scale(zero=False),
                    axis=alt.Axis(title=y_label),
                ),
            )

            # Keep chart size parameters
            chart_width = 950
            chart_height = 500

            # Rows (sections): simple arrows from origin to each section, plus endpoint marker and label.
            section_color = "#e41a1c"
            arrow_df = row_plot_df[[comp_x, comp_y, "label"]].copy()
            arrow_df = arrow_df.rename(columns={comp_x: "x2", comp_y: "y2"})
            arrow_df["x"] = 0.0
            arrow_df["y"] = 0.0

            arrow_chart = (
                alt.Chart(arrow_df)
                .mark_rule(strokeWidth=1.2, opacity=0.7, color=section_color)
                .encode(
                    x="x:Q",
                    y="y:Q",
                    x2="x2:Q",
                    y2="y2:Q",
                    tooltip=[alt.Tooltip("label:N", title=index_col_used)],
                )
            )

            row_chart = alt.Chart(row_plot_df).encode(**base_enc)
            row_points = row_chart.mark_point(
                shape="square", filled=True, size=60, color=section_color
            )
            row_labels = row_chart.mark_text(
                align="left", dx=6, dy=-6, size=14, color=section_color
            ).encode(text=alt.Text("label:N"))

            # Columns (expressions)
            col_chart = alt.Chart(col_plot_df).encode(**base_enc)
            col_points = col_chart.mark_point(filled=True).encode(
                color=alt.Color("タイプ:N", title="タイプ"),
                shape=alt.value("circle"),
                size=alt.Size("freq:Q", title="頻度", scale=alt.Scale(range=[30, 900])),
                tooltip=[
                    alt.Tooltip("label:N", title=expr_col_local),
                    alt.Tooltip("freq:Q", title="頻度"),
                    alt.Tooltip("タイプ:N", title="タイプ"),
                    alt.Tooltip("dominant_category:N", title="dominant_category"),
                    alt.Tooltip(f"{comp_x}:Q", format=".4f"),
                    alt.Tooltip(f"{comp_y}:Q", format=".4f"),
                ],
            )
            col_labels = col_chart.mark_text(align="left", dx=6, dy=6, size=11).encode(
                text=alt.Text("label:N"),
                color=alt.Color("タイプ:N", title="タイプ", legend=None),
            )

            chart = (
                alt.layer(arrow_chart, row_points, row_labels, col_points, col_labels)
                .properties(
                    width=chart_width,
                    height=chart_height,
                    title=f"Correspondence Analysis (Total Inertia: {total_inertia:.4f})",
                )
                .interactive()
                .resolve_scale(color="shared", shape="independent")
            )
            return chart
        except Exception:
            # Fallback to prince's built-in plot if anything goes wrong
            return ca_model.plot(
                ct,
                x_component=0,
                y_component=1,
                show_row_markers=True,
                show_column_markers=True,
                show_row_labels=True,
                show_column_labels=True,
            )


def _get_folder_cache_key(
    folder: Path,
    ext: str | None,
    skip_section_classification: bool | None,
) -> str:
    """Generate a cache key for the folder's content and parsing parameters."""
    # Do not resolve to an absolute path to make the cache portable.
    folder = Path(folder)
    hasher = xxhash.xxh3_64()

    # 1. Parameters that affect parsing
    hasher.update(str(ext).encode())
    hasher.update(str(skip_section_classification).encode())

    # 2. Parsing code version (hash the text_loader module)
    try:
        import dm_annotations.io.text_loader as text_loader_module

        loader_path = Path(inspect.getfile(text_loader_module))
        if loader_path.exists():
            hasher.update(loader_path.read_bytes())
    except (ImportError, TypeError, FileNotFoundError):
        pass  # Continue without code hash if module not found

    # 3. Source file metadata (path, mtime, size)
    files_to_check = []
    if folder.exists():
        # Metadata files
        if (folder / "sources.tsv").exists():
            files_to_check.append(folder / "sources.tsv")
        if (folder / "genres.tsv").exists():
            files_to_check.append(folder / "genres.tsv")

        # Content files
        if ext:
            files_to_check.extend(sorted(folder.glob(f"*{ext}")))
        else:
            files_to_check.extend(sorted(folder.glob("*.md")))
            files_to_check.extend(sorted(folder.glob("*.txt")))

    for f in sorted(files_to_check):
        if f.is_file():
            try:
                stat = f.stat()
                hasher.update(str(f.relative_to(folder)).encode())
                hasher.update(str(stat.st_mtime_ns).encode())
                hasher.update(str(stat.st_size).encode())
            except FileNotFoundError:
                continue  # File might have been deleted during scan

    return hasher.hexdigest()


def load_sections_dataframe(
    folder: Path,
    ext: Optional[str] = None,
    *,
    lazy: bool = True,
    batch_size: int = 2000,
    skip_section_classification: bool | None = True,
    parquet_compression: str | None = "uncompressed",
    prewarm_cache: bool = True,
) -> pl.DataFrame | pl.LazyFrame:
    """
    Load parse_plain_folder_to_tuples output into Polars shards and return a Polars
    LazyFrame by default (or an eager DataFrame when lazy=False).

    This streams parsed sections into small Parquet shards so we do not keep the
    entire corpus in a Python list at once. It preserves full pandoc concurrency
    inside parse_plain_folder_to_tuples but keeps peak memory low.

    :param folder: path to folder with markdown/txt and metadata files
    :param ext: optional, force extension ('.md' or '.txt') or None to auto-detect
    :param lazy: when True return a polars.LazyFrame assembled from parquet shards
    :param batch_size: number of rows per parquet shard (tune to trade IO vs memory)
    :return: Polars LazyFrame (default) or eager DataFrame (lazy=False)
    """

    folder = Path(folder).expanduser().resolve(strict=False)
    logging.info(
        f"Loading sections from folder: {folder} (exists={folder.exists()}, cwd={Path.cwd()})"
    )
    cache_dir = folder / ".cache" / "sections_parquet"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Generate a cache key and check for existing cache file
    cache_key = _get_folder_cache_key(folder, ext, skip_section_classification)
    cache_file = cache_dir / f"sections_{cache_key}.parquet"

    if cache_file.exists():
        logging.info(f"Loading cached sections from {cache_file}")
        if lazy:
            return pl.scan_parquet(str(cache_file))
        else:
            return pl.read_parquet(cache_file)

    # Optionally pre-warm pandoc/pickle cache for markdown files so subsequent parsing is fast.
    if prewarm_cache:
        try:
            md_candidates = []
            if ext is None or ext == ".md":
                md_candidates = sorted(folder.glob("*.md"))
                # fallback to recursive search if top-level empty
                if not md_candidates and folder.exists():
                    md_candidates = list(folder.rglob("*.md"))
            if md_candidates:
                Executor, max_workers = _get_md_parse_executor()
                logging.info(
                    "Pre-warming pandoc/pickle cache for %d markdown files (using %s with %d workers)",
                    len(md_candidates),
                    Executor.__name__,
                    max_workers,
                )

                def _prewarm_one(p: Path) -> None:
                    try:
                        txt = p.read_text(encoding="utf-8")
                        pre_path, pre_txt = _preprocess_and_write_markdown(p, txt)
                        # parse_markdown_blocks_cached will create the .pkl cache file (best-effort)
                        _ = parse_markdown_blocks_cached(pre_path, pre_txt)
                    except Exception:
                        logging.debug("Prewarm failed for %s", p, exc_info=True)

                with Executor(max_workers=max_workers) as exe:
                    # map and exhaust results to trigger work
                    list(exe.map(_prewarm_one, md_candidates))
        except Exception:
            logging.debug("Prewarm cache step failed", exc_info=True)

    rows: list[dict] = []

    # Stream parse (parse_plain_folder_to_tuples is a generator; keeps pandoc concurrency unchanged)
    logging.info("Starting corpus parse")
    for text, meta in tqdm(
        parse_plain_folder_to_tuples(
            folder,
            ext,
            strict_metadata=True,
            section_exclude=SECTION_EXCLUDE_DEFAULT,
            skip_section_classification=skip_section_classification,
        ),
        desc="Streaming docs",
    ):
        # Use normalized taxonomy carried on section meta (from YAML → info → section)
        corpus = (
            meta.get("corpus")
            or meta.get("genre1")
            or (
                meta.get("genre")[0]
                if isinstance(meta.get("genre"), (list, tuple)) and meta.get("genre")
                else meta.get("genre")
            )
        )
        subject_area = meta.get("subject_area") or meta.get("genre2")
        keywords = meta.get("keywords")
        if isinstance(keywords, str):
            try:
                import re as _re

                keywords = [
                    t.strip()
                    for t in _re.split(r"[、,/／;；|\s+]", keywords)
                    if t.strip()
                ]
            except Exception:
                keywords = [keywords.strip()] if keywords and keywords.strip() else None
        elif isinstance(keywords, (list, tuple)):
            keywords = [str(k).strip() for k in keywords if str(k).strip()]
        else:
            keywords = None
        corpus = str(corpus).strip() if corpus else None
        subject_area = str(subject_area).strip() if subject_area else None

        genre = corpus
        genre1 = corpus
        genre2 = subject_area
        genre3 = keywords[0] if isinstance(keywords, list) and keywords else None
        genre_path = " / ".join(s for s in [corpus, subject_area] if s)

        # Debug: enforce and log taxonomy presence per section
        if not genre1 or not genre2:
            logging.error(
                "Missing taxonomy in load_sections_dataframe: basename=%s title=%s section=%s corpus=%r subject_area=%r",
                meta.get("basename"),
                meta.get("title"),
                meta.get("section"),
                corpus,
                subject_area,
            )
            logging.error(
                "Meta snapshot: %s",
                {
                    k: meta.get(k)
                    for k in (
                        "corpus",
                        "subject_area",
                        "genre",
                        "genre1",
                        "genre2",
                        "genre3",
                        "genre_path",
                        "keywords",
                    )
                },
            )
            assert genre1, (
                f"load_sections_dataframe: genre1 missing for basename {meta.get('basename')}"
            )
            assert genre2, (
                f"load_sections_dataframe: genre2 missing for basename {meta.get('basename')}"
            )

        rows.append(
            {
                "basename": str(meta.get("basename", "") or ""),
                "title": str(meta.get("title", "") or ""),
                "section": str(meta.get("section") or "unknown"),
                "text": text or "",
                "text_len": len(text or ""),
                "year": meta.get("year"),
                "genre": genre,
                "genre1": genre1,
                "genre2": genre2,
                "genre3": genre3,
                "corpus": corpus,
                "subject_area": subject_area,
                "keywords": keywords,
                "genre_path": genre_path if genre_path else None,
                "author": meta.get("author") or None,
                "section_subtypes": meta.get("section_subtypes") or None,
                "section_matched_subtypes": meta.get("section_matched_subtypes")
                or None,
                "section_number": meta.get("section_number"),
                "section_level": meta.get("section_level"),
                "section_parent": meta.get("section_parent"),
                "section_path": meta.get("section_path"),
            }
        )

    if not rows:
        logging.warning("Nothing was parsed!")
        # Nothing parsed
        return pl.DataFrame([])

    df_all = pl.DataFrame(rows)
    df_all.write_parquet(cache_file, compression=parquet_compression)

    if lazy:
        return pl.scan_parquet(str(cache_file))
    else:
        return df_all


def summarize_by_file(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    """
    Compute per-file summary: number of sections, unknown count, and unknown ratio.

    This function accepts either a Polars eager DataFrame or a Polars LazyFrame.
    When a LazyFrame is provided a lazy groupby/aggregation is performed and only
    the aggregated result is collected into memory (avoids materializing the
    full dataset). When an eager DataFrame is provided the existing logic is used.

    Returned columns:
      - basename, n_sections, n_unknown, unknown_ratio, title, example_text, year
    """
    # If caller passed a LazyFrame, perform the aggregation lazily and collect only the small result.
    if isinstance(df, pl.LazyFrame):
        # Build lazy aggregation expressions
        agg_exprs = [
            pl.col("section").len().cast(pl.Int64).alias("n_sections"),
            (pl.col("section") == "unknown").cast(pl.Int64).sum().alias("n_unknown"),
            pl.col("title").first().alias("title"),
            pl.col("text").first().alias("example_text"),
            pl.col("year").first().alias("year"),
        ]
        # Use groupby on the lazy frame, then compute unknown_ratio lazily before collecting.
        summary_lf = (
            df.group_by("basename")
            .agg(agg_exprs)
            .with_columns(
                pl.when(pl.col("n_sections") > 0)
                .then(pl.col("n_unknown") / pl.col("n_sections"))
                .otherwise(0.0)
                .alias("unknown_ratio")
            )
        )
        summary = summary_lf.collect()
        return summary

    # Eager DataFrame path (existing behavior)
    if df.is_empty():
        return pl.DataFrame(
            schema={
                "basename": pl.Utf8,
                "n_sections": pl.Int64,
                "n_unknown": pl.Int64,
                "unknown_ratio": pl.Float64,
                "title": pl.Utf8,
                "example_text": pl.Utf8,
                "year": pl.Int64,
            }
        )

    summary = (
        df.group_by("basename")
        .agg(
            [
                # total number of sections per file
                pl.col("section").len().cast(pl.Int64).alias("n_sections"),
                # number of 'unknown' sections
                (pl.col("section") == "unknown")
                .cast(pl.Int64)
                .sum()
                .alias("n_unknown"),
                pl.col("title").first().alias("title"),
                pl.col("text").first().alias("example_text"),
                pl.col("year").first().alias("year"),
            ]
        )
        .with_columns(
            # guard division by zero, produce float ratio
            pl.when(pl.col("n_sections") > 0)
            .then(pl.col("n_unknown") / pl.col("n_sections"))
            .otherwise(0.0)
            .alias("unknown_ratio")
        )
    )
    return summary


def sample_by_section(
    df: pl.DataFrame | pl.LazyFrame,
    max_rows_per_section: int,
    section_col: str = "section",
    balance_on: str | None = None,
    seed: int = 42,
) -> pl.DataFrame:
    """
    Deterministically sample each section to at most max_rows_per_section rows.

    Accepts either a Polars DataFrame or LazyFrame. For LazyFrame inputs, only
    distinct section names are collected first, and then each section partition
    is collected independently to keep peak memory bounded.

    If `balance_on` is provided, it performs stratified sampling to get as close
    as possible to `max_rows_per_section`. It samples an equal base number of rows
    from each subgroup (defined by unique values in the `balance_on` column) and
    randomly distributes the remainder of rows across the subgroups.

    Args:
        df: Input DM/sections dataframe (eager or lazy).
        max_rows_per_section: Maximum number of rows to retain per unique section.
        section_col: Column name defining the section grouping (default: "section").
        balance_on: Optional column name to balance sampling over within each section.
        seed: Random seed for deterministic sampling.

    Returns:
        A Polars DataFrame containing the sampled rows per section.

    Examples:
    >>> import polars as pl
    >>> df = pl.DataFrame({"section": ["A"]*5 + ["B"]*3, "x": list(range(8))})
    >>> out = sample_by_section(df, max_rows_per_section=2, section_col="section", seed=0)
    >>> set(out["section"]) == {"A","B"} and out.group_by("section").len().select("len").to_series().to_list() == [2,2]
    True
    """
    if max_rows_per_section <= 0:
        return pl.DataFrame([])

    def _sample_partition(part_df: pl.DataFrame) -> pl.DataFrame:
        if part_df.height <= max_rows_per_section:
            return part_df

        if balance_on and balance_on in part_df.columns:
            groups = part_df.group_by(balance_on).count()
            n_groups = groups.height
            if n_groups > 0:
                n_per_group = max_rows_per_section // n_groups
                remainder = max_rows_per_section % n_groups

                # Determine which groups get an extra sample
                group_names = groups.get_column(balance_on).to_list()
                rng = random.Random(seed)
                groups_with_extra = set(rng.sample(group_names, remainder))

                # Create a mapping of group -> target_size
                target_sizes = {
                    name: n_per_group + 1 if name in groups_with_extra else n_per_group
                    for name in group_names
                }

                # Create a mapping DataFrame to join
                target_df = pl.DataFrame(
                    {
                        balance_on: group_names,
                        "target_size": [target_sizes[name] for name in group_names],
                    }
                )

                # Join the target sizes to the partition DataFrame
                part_df_with_target = part_df.join(target_df, on=balance_on, how="left")

                # Use window function for grouped sampling
                return (
                    part_df_with_target.with_columns(
                        pl.int_range(0, pl.count())
                        .shuffle(seed=seed)
                        .over(balance_on)
                        .alias("int_range")
                    )
                    .filter(pl.col("int_range") < pl.col("target_size"))
                    .drop(["int_range", "target_size"])
                )

        # Fallback to simple sampling
        return part_df.sample(n=max_rows_per_section, seed=seed)

    sampled: list[pl.DataFrame] = []
    is_lazy = isinstance(df, pl.LazyFrame)

    if is_lazy:
        sections = (
            df.select(pl.col(section_col)).unique().collect()[section_col].to_list()
        )
    else:
        sections_ser = df.select(section_col).unique()[section_col]
        sections = (
            sections_ser.to_list()
            if hasattr(sections_ser, "to_list")
            else list(sections_ser)
        )

    for sec in sections:
        part_df = df.filter(pl.col(section_col) == sec)
        if is_lazy:
            part_df = part_df.collect()

        sampled.append(_sample_partition(part_df))

    if not sampled:
        return pl.DataFrame([])

    return pl.concat(sampled, how="vertical")


def plot_section_distribution(summary_df: pl.DataFrame) -> alt.Chart:
    """
    Produce a three-panel Altair visualization:
      - Top: histogram of number of sections per file
      - Middle: histogram of unknown_ratio distribution
      - Bottom: scatter of n_sections vs unknown_ratio with tooltips

    Returns an Altair Chart object that displays in a Jupyter notebook.

    Example:
    >>> import polars as pl
    >>> tmp = pl.DataFrame({'basename':['a','b','c'],'n_sections':[1,3,2],'n_unknown':[1,0,1],'unknown_ratio':[1.0,0.0,0.5]})
    >>> isinstance(plot_section_distribution(tmp), alt.Chart)
    True
    """
    pdf = summary_df.select(
        ["basename", "n_sections", "n_unknown", "unknown_ratio"]
    ).to_pandas()

    rng = random.Random(42)
    jitter_x = 0.2
    n_rows = len(pdf)
    if n_rows > 0:
        pdf["_n_sections_jitter"] = pdf["n_sections"].astype(float) + [
            rng.uniform(-jitter_x, jitter_x) for _ in range(n_rows)
        ]
        # Ensure jittered positions are not negative
        pdf["_n_sections_jitter"] = pdf["_n_sections_jitter"].clip(lower=0.0)

    hist = (
        alt.Chart(pdf)
        .mark_bar()
        .encode(
            x=alt.X("n_sections:O", title="Number of sections per file"),
            y=alt.Y("count()", title="Number of files"),
            tooltip=[alt.Tooltip("count()", title="files")],
        )
        .properties(width=700, height=200)
    )

    hist_unknown = (
        alt.Chart(pdf)
        .mark_bar()
        .encode(
            x=alt.X(
                "unknown_ratio:Q",
                bin=alt.Bin(step=0.05),
                title="Unknown ratio",
            ),
            y=alt.Y("count()", title="Number of files"),
            tooltip=[alt.Tooltip("count()", title="files")],
        )
        .properties(width=700, height=200)
    )

    scatter = (
        alt.Chart(pdf)
        .mark_circle(opacity=0.8, size=60)
        .encode(
            x=alt.X("_n_sections_jitter:Q", title="Number of sections (jittered)"),
            y=alt.Y("unknown_ratio:Q", title="Unknown ratio"),
            color=alt.Color(
                "unknown_ratio:Q", title="Unknown ratio", scale=alt.Scale(scheme="reds")
            ),
            tooltip=[
                "basename",
                "n_sections",
                "n_unknown",
                alt.Tooltip("unknown_ratio", format=".2f"),
            ],
        )
        .properties(width=700, height=300)
    )

    # Stack the three panels vertically
    return alt.vconcat(hist, hist_unknown, scatter).resolve_scale(color="independent")


def find_problematic(
    summary_df: pl.DataFrame,
    min_sections: int = 2,
    min_unknown_ratio: float = 0.6,
    top_k: int = 200,
) -> pl.DataFrame:
    """
    Return files that are likely problematic. Criteria:
      - Files with n_sections <= min_sections
      - Files with unknown_ratio >= min_unknown_ratio

    Results are deduplicated and sorted by unknown_ratio desc, n_sections asc.

    :param summary_df: output of summarize_by_file
    :param min_sections: threshold for "too few sections"
    :param min_unknown_ratio: threshold for "mostly unknown"
    :param top_k: maximum rows to return

    :return: Polars DataFrame with snippet column added

    Example:
    >>> df = pl.DataFrame([{'basename':'a','n_sections':1,'n_unknown':1,'unknown_ratio':1.0,'title':'T1','example_text':'longtext'},{'basename':'b','n_sections':5,'n_unknown':0,'unknown_ratio':0.0,'title':'T2','example_text':'ok'}])
    >>> find_problematic(df,min_sections=2,min_unknown_ratio=0.5).to_dicts()[0]['basename'] in {'a'}
    True
    """
    low_sections = summary_df.filter(pl.col("n_sections") <= min_sections)
    high_unknown = summary_df.filter(pl.col("unknown_ratio") >= min_unknown_ratio)
    combined = pl.concat([low_sections, high_unknown]).unique(subset="basename")
    combined = combined.with_columns(
        pl.col("example_text").str.slice(0, 320).alias("snippet")
    )
    combined = combined.sort(["unknown_ratio", "n_sections"], descending=[True, False])
    return combined.head(top_k)


def _highlight_segments(sentence: str, intervals: list[tuple[int, int, str]]) -> str:
    """
    Build HTML for a sentence with highlighted intervals.

    Intervals are (start, end, kind) in sentence-relative character indices.
    kind is one of {"接続表現","文末表現"}.
    """
    if not sentence:
        return ""
    n = len(sentence)
    # Clamp and normalize intervals
    norm: list[tuple[int, int, str]] = []
    for s, e, k in intervals:
        if s is None or e is None:
            continue
        s0 = max(0, min(n, int(s)))
        e0 = max(0, min(n, int(e)))
        if e0 <= s0:
            continue
        norm.append((s0, e0, k))
    if not norm:
        return html.escape(sentence)

    boundaries = sorted({0, n, *[s for s, _, _ in norm], *[e for _, e, _ in norm]})
    pieces: list[str] = []
    for i in range(len(boundaries) - 1):
        a, b = boundaries[i], boundaries[i + 1]
        if a == b:
            continue
        # Determine which kinds cover this segment
        kinds = set()
        for s, e, k in norm:
            if s < b and e > a:
                kinds.add(k)
        seg_text = html.escape(sentence[a:b])
        if not kinds:
            pieces.append(seg_text)
        else:
            has_conn = "接続表現" in kinds
            has_end = "文末表現" in kinds
            cls = (
                "hl-both"
                if has_conn and has_end
                else ("hl-conn" if has_conn else "hl-end")
            )
            pieces.append(f'<span class="{cls}">{seg_text}</span>')
    return "".join(pieces)


def collocation_sentences_html(
    df: pl.DataFrame,
    query: str = "",
    case_sensitive: bool = False,
    max_rows: int = 200,
    conn_col: str | None = None,
    conn_value: str | None = None,
    end_col: str | None = None,
    end_value: str | None = None,
    require_both: bool = True,
    show_paragraph: bool = False,
    all_sentences_df: pl.DataFrame | None = None,
) -> str:
    """
    Render an HTML listing of sentences that contain collocations (both 接続表現 and 文末表現).

    Two modes:
      - DM selector mode (preferred): pass conn_col/conn_value and/or end_col/end_value to
        select target expressions by raw 表現 or 機能_* columns. Sentences must contain both types;
        if only one side is specified, the other may be any expression of that type.
        Highlights show DM spans (as before).
      - Regex mode: if query is provided and no selector values are given, filter sentences whose
        sentence_text matches the regex and highlight the matched substrings (not DM spans).

    Args:
        df: DM matches with sentence metadata.
        query: Regex to filter sentence_text. Used only when no selector values are provided.
        case_sensitive: Regex case sensitivity for query.
        max_rows: Maximum number of sentences to render.
        conn_col: Column to filter 接続表現 on ("表現" or 機能_*). None means no constraint.
        conn_value: Selected value for conn_col. None means no constraint.
        end_col: Column to filter 文末表現 on ("表現" or 機能_*). None means no constraint.
        end_value: Selected value for end_col. None means no constraint.
        require_both: When True, sentences must contain at least one 接続表現 and one 文末表現.
        show_paragraph: When True, show the full paragraph with the sentence highlighted.
        all_sentences_df: Optional DataFrame containing all sentences (one row per sentence)
                          to enable searching sentences that do not have any DM matches.

    Returns:
        HTML string for rendering in a notebook.
    """
    if df is None:
        df = pl.DataFrame()

    if all_sentences_df is not None and not all_sentences_df.is_empty():
        dm_cols = [
            "sentence_id",
            "タイプ",
            "表現",
            "機能",
            "細分類",
            "position",
            "end_position",
            "span_text",
        ]
        dm_cols_present = [c for c in dm_cols if c in df.columns]
        df_dms = df.select(dm_cols_present)
        # Use a left join to keep all sentences, with nulls for DM columns in sentences without DMs.
        work_base = all_sentences_df.join(df_dms, on="sentence_id", how="left")
    else:
        work_base = df

    if work_base.is_empty():
        return "<div class='colloc-root'><em>No data.</em></div>"

    # Calculate total sentences in the current scope for PPM calculation
    total_sentences = work_base.select(pl.col("sentence_id")).n_unique()

    # Ensure required columns exist
    req = {"sentence_id", "sentence_text"}
    missing = req - set(work_base.columns)
    if missing:
        raise ValueError(
            f"DataFrame missing required columns for collocation view: {sorted(missing)}"
        )

    # Base frame
    work = work_base

    # Optional sentence-level regex filtering if query is provided
    if query:
        if case_sensitive:
            work = work.filter(pl.col("sentence_text").str.contains(query))
        else:
            work = (
                work.with_columns(
                    pl.col("sentence_text").str.to_lowercase().alias("_sent_l")
                )
                .filter(pl.col("_sent_l").str.contains(query.lower()))
                .drop("_sent_l")
            )

    # Determine mode
    selector_mode = (conn_value is not None) or (end_value is not None)
    regex_mode = (not selector_mode) and bool(query)

    # Compute sentence set
    sent_ids: list[int] = []

    if selector_mode:
        # Validate selector columns if provided
        if conn_value is not None and conn_col is None:
            conn_col = "表現"
        if end_value is not None and end_col is None:
            end_col = "表現"

        if conn_col is not None and conn_col not in df.columns:
            raise ValueError(f"Selected conn_col '{conn_col}' not found in DataFrame")
        if end_col is not None and end_col not in df.columns:
            raise ValueError(f"Selected end_col '{end_col}' not found in DataFrame")

        # Precompute sentence id sets for any connective / any sentence-final presence
        sids_with_any_conn = set(
            work.filter(pl.col("タイプ") == "接続表現")
            .select("sentence_id")
            .unique()
            .to_series()
        )
        sids_with_any_end = set(
            work.filter(pl.col("タイプ") == "文末表現")
            .select("sentence_id")
            .unique()
            .to_series()
        )

        # For all cases, apply constraints from both sides and combine.
        all_sids = set(work.select("sentence_id").unique().to_series())

        # Determine connective sentence set
        if conn_value is None:  # any: requires at least one connective
            sids_conn = sids_with_any_conn
        elif conn_value == "(None)":  # none: requires no connectives
            sids_conn = all_sids - sids_with_any_conn
        else:  # specific
            cond = (pl.col("タイプ") == "接続表現") & (
                pl.col(conn_col) == conn_value
            )
            sids_conn = set(
                work.filter(cond).select("sentence_id").unique().to_series()
            )

        # Determine end-form sentence set
        if end_value is None:  # any: requires at least one end-form
            sids_end = sids_with_any_end
        elif end_value == "(None)":  # none: requires no end-forms
            sids_end = all_sids - sids_with_any_end
        else:  # specific
            cond = (pl.col("タイプ") == "文末表現") & (pl.col(end_col) == end_value)
            sids_end = set(
                work.filter(cond).select("sentence_id").unique().to_series()
            )

        # Combine the sets by intersection.
        # The case of (any, any) is handled outside this `if selector_mode` block.
        sids_to_filter = sids_conn & sids_end

        sent_ids = sorted(list(sids_to_filter))
    else:
        # No selectors -> default collocation constraint depends on require_both
        if require_both:
            type_counts = (
                work.select(["sentence_id", "タイプ"])
                .unique()
                .group_by("sentence_id")
                .agg(pl.col("タイプ").n_unique().alias("_ntypes"))
                .filter(pl.col("_ntypes") >= 2)
                .sort("sentence_id")
            )

            # If there are no collocations, we still want to report zero hits (after possible regex filter)
            if type_counts.is_empty():
                sent_ids = []
            else:
                sent_ids = type_counts["sentence_id"].to_list()
        else:
            # If not requiring both, just get all unique sentence IDs from the (potentially query-filtered) work df
            sent_ids = work.select("sentence_id").unique().to_series().to_list()

    # Deduplicate sentences before applying max_rows
    if sent_ids:
        sids_order_df = pl.DataFrame(
            {"sentence_id": sent_ids, "_order": range(len(sent_ids))}
        )
        sent_meta_df = (
            work.select(["sentence_id", "basename", "section", "sentence_text"])
            .unique(subset=["sentence_id"])
            .join(sids_order_df, on="sentence_id", how="inner")
            .sort("_order")
            .unique(subset=["basename", "section", "sentence_text"], keep="first")
        )
        sent_ids = sent_meta_df["sentence_id"].to_list()

    n_hits = len(sent_ids)
    ppm = (n_hits / total_sentences) * 1_000_000 if total_sentences > 0 else 0.0

    # Enforce max_rows for display
    sent_ids_display = sent_ids[: max_rows if max_rows > 0 else 0]

    # --- Build HTML ---
    css = """
<style>
.colloc-root { font-family: system-ui, -apple-system, Segoe UI, Roboto, Noto Sans, Helvetica, Arial, sans-serif; }
.colloc-summary { font-size: 14px; color: #475569; margin-bottom: 10px; padding: 6px 0; border-bottom: 1px solid #e5e7eb; }
.colloc-item { padding: 10px 12px; margin: 8px 0; border: 1px solid #e5e7eb; border-radius: 8px; background: #fff; }
.colloc-meta { color: #334155; font-size: 13px; margin-bottom: 6px; display: flex; gap: 12px; flex-wrap: wrap; }
.colloc-meta .label { font-weight: 600; color: #111827; }
.sentence { font-size: 16px; line-height: 1.7; color: #0f172a; word-break: break-word; white-space: pre-wrap; }
.hl-conn { background-color: #fff3c4; border-radius: 2px; }
.hl-end { background-color: #c7ebff; border-radius: 2px; }
.hl-both { background-image: linear-gradient(90deg, #fff3c4, #c7ebff); border-radius: 2px; }
.hl-sentence { text-decoration: underline; }
.expr-list { font-size: 13px; color: #374151; margin-top: 4px; }
.expr-list .expr { font-weight: 600; }
.legend { font-size: 12px; color: #475569; margin: 8px 0; }
.legend .box { display: inline-block; width: 10px; height: 10px; vertical-align: middle; margin-right: 6px; border-radius: 2px; }
.legend .conn { background: #fff3c4; }
.legend .end { background: #c7ebff; }
.legend .both { background-image: linear-gradient(90deg, #fff3c4, #c7ebff); }
</style>
"""
    displaying_msg = ""
    if n_hits > len(sent_ids_display):
        displaying_msg = f"Displaying first {len(sent_ids_display)}."
    elif n_hits > 0 and len(sent_ids_display) == 0:
        displaying_msg = f"Not displaying any rows (max_rows is {max_rows})."

    summary_html = f"""
<div class="colloc-summary">
  Found <strong>{n_hits}</strong> matching sentences ({ppm:.2f} per million sentences).
  {displaying_msg}
</div>
"""

    legend_html = """
<div class="legend">
  <span class="box conn"></span>接続表現　
  <span class="box end"></span>文末表現　
  <span class="box both"></span>重なり
</div>
"""
    html_prefix = f'<div class="colloc-root">{summary_html}{legend_html}'

    if n_hits == 0:
        return '<div class="colloc-root"><em>No matching sentences.</em></div>'

    if not sent_ids_display:
        return css + html_prefix + "</div>"

    # Restrict to selected sentences for display
    work = work.filter(pl.col("sentence_id").is_in(sent_ids_display))
    items: list[str] = []

    # Regex matcher for highlighting when in regex mode
    if regex_mode:
        import re as _re

        flags = 0 if case_sensitive else _re.IGNORECASE
        try:
            rx = _re.compile(query, flags)
        except Exception:
            return css + html_prefix + "<em>Invalid regex.</em></div>"
    else:
        rx = None  # type: ignore

    # Iterate per sentence_id
    for sid in sent_ids_display:
        rows = work.filter(pl.col("sentence_id") == sid)
        if rows.is_empty():
            continue
        r0 = rows.row(0, named=True)
        sentence = r0.get("sentence_text") or ""
        title = r0.get("title") or ""
        genre2 = r0.get("genre2") or ""
        section = r0.get("section") or ""
        genre = r0.get("ジャンル") or ""
        basename = r0.get("basename") or ""
        year = r0.get("year")

        # Collect highlights and expression lists
        intervals: list[tuple[int, int, str]] = []
        conn_exprs: set[str] = set()
        end_exprs: set[str] = set()

        if regex_mode and rx is not None:
            # Highlight regex matches instead of DM spans
            for m in rx.finditer(str(sentence)):
                intervals.append(
                    (m.start(), m.end(), "文末表現")
                )  # reuse blue highlight
            # Also summarize expressions present in the sentence (without affecting highlight)
            for rec in rows.select(["タイプ", "表現"]).unique().to_dicts():
                t = rec.get("タイプ")
                e = rec.get("表現")
                if t == "接続表現":
                    conn_exprs.add(str(e))
                elif t == "文末表現":
                    end_exprs.add(str(e))
        else:
            # Highlight DM spans (as before)
            for rec in rows.to_dicts():
                t = rec.get("タイプ")
                expr = rec.get("表現")
                pos = rec.get("position")
                end_pos = rec.get("end_position")
                span_text = rec.get("span_text") or ""
                if pos is None:
                    continue
                try:
                    start_rel = int(pos)
                    if end_pos is None:
                        end_rel = start_rel + len(span_text)
                    else:
                        end_rel = int(end_pos)
                except (ValueError, TypeError):
                    continue
                if isinstance(t, str):
                    if t == "接続表現":
                        conn_exprs.add(str(expr))
                    elif t == "文末表現":
                        end_exprs.add(str(expr))
                intervals.append((start_rel, end_rel, t if isinstance(t, str) else ""))

        highlighted_sentence = _highlight_segments(str(sentence), intervals)
        display_text: str
        if show_paragraph:
            paragraph_text = r0.get("paragraph_text") or ""
            # To highlight the sentence within the paragraph, we can wrap it.
            # The DM highlights are already inside `highlighted_sentence`.
            highlighted_sentence_in_para = (
                f"<span class='hl-sentence'>{highlighted_sentence}</span>"
            )

            # Escape the full paragraph, then replace the escaped sentence with our HTML version.
            escaped_paragraph = html.escape(paragraph_text)
            escaped_sentence = html.escape(sentence)

            if escaped_sentence in escaped_paragraph:
                display_text = escaped_paragraph.replace(
                    escaped_sentence, highlighted_sentence_in_para, 1
                )
            else:
                # Fallback if sentence isn't found (e.g., whitespace differences)
                display_text = highlighted_sentence_in_para
        else:
            display_text = highlighted_sentence

        meta_parts = []
        if title:
            title_disp = html.escape(str(title))
            if basename:
                title_disp = f"{title_disp} ({html.escape(str(basename))})"
            meta_parts.append(
                f"<span><span class='label'>Title:</span> {title_disp}</span>"
            )
        if genre2:
            meta_parts.append(
                f"<span><span class='label'>Genre2:</span> {html.escape(str(genre2))}</span>"
            )
        if section:
            meta_parts.append(
                f"<span><span class='label'>Section:</span> {html.escape(str(section))}</span>"
            )
        if genre:
            meta_parts.append(
                f"<span><span class='label'>Genre:</span> {html.escape(str(genre))}</span>"
            )
        if basename:
            meta_parts.append(
                f"<span><span class='label'>File:</span> {html.escape(str(basename))}</span>"
            )
        if year is not None:
            meta_parts.append(
                f"<span><span class='label'>Year:</span> {html.escape(str(year))}</span>"
            )

        conn_list = "、".join(sorted(e for e in conn_exprs if e))
        end_list = "、".join(sorted(e for e in end_exprs if e))
        expr_html = (
            f"<div class='expr-list'>"
            f"<span class='label'>接続表現:</span> <span class='expr'>{html.escape(conn_list)}</span>　"
            f"<span class='label'>文末表現:</span> <span class='expr'>{html.escape(end_list)}</span>"
            f"</div>"
        )

        item_html = (
            "<div class='colloc-item'>"
            f"<div class='colloc-meta'>{' '.join(meta_parts)}</div>"
            f"<div class='sentence'>{display_text}</div>"
            f"{expr_html}"
            "</div>"
        )
        items.append(item_html)

    if not items:
        return '<div class="colloc-root"><em>No matching sentences.</em></div>'

    if not sent_ids_display:
        return css + html_prefix + "</div>"

    return css + html_prefix + "\n".join(items) + "\n</div>"
