import gc
import logging
import os
import pickle
import random
import re
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path
from typing import Any, Iterator, Literal

import orjson
import pandoc
import polars as pl
import xxhash
import yaml
from pandoc.types import (
    BlockQuote,
    BulletList,
    Div,
    Header,
    OrderedList,
    Para,
    Plain,
    Space,
    Str,
)
from spacy.language import Language
from spacy.tokens import Doc
from wtpsplit import SaT

from dm_annotations.io.loader import CorpusParser, normalize_meta


@dataclass
class SectionNode:
    level: int
    num: int | None
    num_prefix: str | None
    raw_text: str
    start_char: int
    end_char: int | None = None
    category: set[str] | None = None
    subtypes: set[str] | None = None
    matched_subtypes: set[str] | None = None
    children: list["SectionNode"] = field(default_factory=list)
    synthetic: bool = False
    block: Any = None
    block_idx: int | None = None


def _process_md_file(args: tuple[Path, dict[str, Any], set[str], bool]):
    """Process a single Markdown file into (plain_text, meta_dict) tuples."""
    # Unpack args: (path, info, section_exclude, strict_metadata, skip_section_classification)
    try:
        (
            txt_path,
            info,
            section_exclude,
            strict_metadata,
            skip_section_classification,
        ) = args
    except ValueError:
        try:
            txt_path, info, section_exclude, strict_metadata = args
            skip_section_classification = None
        except ValueError:
            # back-compat: accept older 3-tuple tasks
            txt_path, info, section_exclude = args
            strict_metadata = True
            skip_section_classification = None

    if info is None:
        info = {}

    try:
        original_text = txt_path.read_text(encoding="utf-8")
    except Exception as e:
        logging.warning(f"Could not read {txt_path}: {e}")
        return []

    pre_path, pre_text = _preprocess_and_write_markdown(txt_path, original_text)

    yaml_info: dict[str, Any] | None = None
    try:
        yaml_meta = _parse_yaml_front_matter(original_text) or _parse_yaml_front_matter(
            pre_text
        )
        if yaml_meta:
            try:
                yaml_info = _yaml_meta_to_info(yaml_meta, txt_path)
            except Exception:
                logging.debug(
                    "Failed to convert YAML front matter for %s",
                    txt_path,
                    exc_info=True,
                )
    except Exception:
        yaml_meta = None

    # Normalize only the non-YAML portion, then overlay YAML so its fields are preserved.
    try:
        base_info = normalize_meta(info or {})
    except Exception:
        base_info = info or {}
    info = {**base_info, **(yaml_info or {})}
    info = normalize_genre_meta(info)

    # Enforce strict metadata if requested (raise to propagate to caller)
    if strict_metadata:
        if not (info.get("journal_title") or info.get("title")):
            raise ValueError(f"Missing title for basename {info.get('basename')}")
        # Year is optional now — log debug when missing/invalid but do not abort.
        if not info.get("year") or not str(info.get("year")).isdigit():
            logging.debug(
                "Missing or invalid year for basename %s: %s -- continuing without year",
                info.get("basename"),
                info.get("year"),
            )
        # require a genre; YAML produces a 3-element list under 'genre' so this works both ways
        if not info.get("genre"):
            raise ValueError(f"Missing genre for basename {info.get('basename')}")

    try:
        blocks, sanitized_text = parse_markdown_blocks_cached(pre_path, pre_text)
    except Exception as e:
        raise RuntimeError(
            f"Failed to parse markdown {txt_path} with pandoc: {e}"
        ) from e

    # Prefer document title from YAML; fall back to journal title/name
    title = (
        info.get("title") or info.get("journal_title") or info.get("journal_name") or ""
    )
    if (
        len(blocks) >= 2
        and isinstance(blocks[0], Header)
        and blocks[0][0] == 1
        and isinstance(blocks[1], Header)
        and blocks[1][0] == 2
    ):

        def extract_text_from_inlines(inlines):
            text_parts = []
            for inline in inlines:
                if isinstance(inline, Str):
                    text_parts.append(inline[0])
                elif isinstance(inline, Space):
                    text_parts.append(" ")
                elif hasattr(inline, "__iter__") and len(inline) > 0:
                    try:
                        nested_inlines = inline[-1]
                        if isinstance(nested_inlines, list):
                            text_parts.append(extract_text_from_inlines(nested_inlines))
                    except (IndexError, TypeError):
                        pass
            return "".join(text_parts)

        first_h1 = remove_cjk_whitespace(
            extract_text_from_inlines(blocks[0][2]).strip()
        )
        first_h2 = remove_cjk_whitespace(
            extract_text_from_inlines(blocks[1][2]).strip()
        )
        # Prefer the document's first H1/H2 as the title/subtitle for parsing so
        # they will be treated as metadata rather than content.
        title = first_h1
        info["subtitle"] = first_h2
        info["detected_title"] = first_h1

    real_nodes = extract_section_nodes(
        blocks,
        title,
        info.get("subtitle"),
        sanitized_text,
        skip_classification=skip_section_classification,
    )

    # Determine annotated mode: explicit flag or auto-detect Header Attrs
    annotated_mode = bool(
        (skip_section_classification is True) or _blocks_have_header_attrs(blocks)
    )

    if annotated_mode:
        # Respect pandoc-assigned header levels; do not synthesize or normalize numbering
        all_nodes = list(real_nodes)
    else:
        # Original numbering/normalization flow
        # Use the sanitized text (with <br> removed) for synthetic-node detection and for all slicing.
        all_nodes = merge_synthetic_nodes(real_nodes, sanitized_text)
        # Assign hierarchical numbering for inline enumerations such as "仮説 1: ..."
        _assign_enumerated_hypotheses(all_nodes)
        _normalize_numeric_sequences(all_nodes)
        _promote_numeric_section_prefixes(all_nodes, base_level=2)
        _smooth_level_jumps(all_nodes)

    # Force explicit "references" headings to be top-level (canonical base level 2).
    # Some documents use deep header levels for references; treat them as section anchors
    # so they appear as siblings rather than children of the preceding section.
    if not annotated_mode:
        try:
            for n in all_nodes:
                if (
                    getattr(n, "matched_subtypes", None)
                    and "references" in n.matched_subtypes
                ):
                    if (n.level or 0) != 2:
                        logging.debug(
                            "Promoting 'references' heading to level 2: start=%s raw=%r",
                            getattr(n, "start_char", None),
                            getattr(n, "raw_text", None),
                        )
                        n.level = 2
        except Exception:
            logging.debug("Failed to promote 'references' headings", exc_info=True)

    root = build_section_tree(all_nodes, len(sanitized_text))

    # Lift canonical top-level sections (references, appendix, toc, abstract, notes, etc.)
    # out to the document root so they are not nested under preceding sections.
    if not annotated_mode:
        TOP_LEVEL_SECTIONS = {
            "references",
            "appendix",
            "toc",
            "abstract",
            "acknowledgments",
            "author_bio",
            "notes",
        }
        try:
            moved: list[SectionNode] = []

            def _collect_and_lift(parent: SectionNode) -> None:
                for ch in list(parent.children):
                    ms = getattr(ch, "matched_subtypes", None)
                    if ms and set(ms) & TOP_LEVEL_SECTIONS:
                        parent.children.remove(ch)
                        ch.level = 2
                        moved.append(ch)
                    else:
                        _collect_and_lift(ch)

            _collect_and_lift(root)
            if moved:
                root.children.extend(moved)
                root.children.sort(key=lambda n: n.start_char or 0)
                # recompute end_char ranges after structural changes
                assign_end_chars(root, len(sanitized_text))
        except Exception:
            logging.debug("Failed to lift canonical top-level sections", exc_info=True)

    _propagate_child_categories(root)
    collapse_same_category(root)

    output = []
    for plain_text, meta_dict in yield_section_texts(
        root, info, title, section_exclude, sanitized_text
    ):
        output.append((plain_text, meta_dict))
    return output


def _flatten_table_to_paragraphs(table_block: Any) -> list[Any] | None:
    """
    If table_block looks like a simple two-column "newspaper" table, return a list
    of Para blocks representing the left-column then right-column concatenations.
    Return None if the block is not a flattenable two-column flow table.

    Heuristics:
      - finds a nested list-of-rows structure inside the table AST
      - requires exactly 2 columns and at least 3 rows
      - both combined column texts must be substantial and mostly-CJK
    """

    def _find_rows(obj, seen: set[int] | None = None):
        if seen is None:
            seen = set()
        try:
            oid = id(obj)
        except Exception:
            oid = None
        if oid in seen:
            return None
        if oid is not None:
            seen.add(oid)

        # straightforward case: a list-of-rows (each row is a list)
        if isinstance(obj, list) and obj and all(isinstance(r, list) for r in obj):
            counts = {len(r) for r in obj if isinstance(r, list)}
            if len(counts) == 1 and next(iter(counts)) >= 2:
                return obj

        # recurse into iterables/containers
        if isinstance(obj, (list, tuple)):
            for e in obj:
                found = _find_rows(e, seen)
                if found:
                    return found
        else:
            try:
                for e in list(obj):
                    found = _find_rows(e, seen)
                    if found:
                        return found
            except Exception:
                pass
        return None

    def _cell_text(cell: Any) -> str:
        if isinstance(cell, list):
            return remove_cjk_whitespace(_extract_text_from_blocks(cell)).strip()
        try:
            return remove_cjk_whitespace(_extract_text_from_blocks([cell])).strip()
        except Exception:
            try:
                return str(cell).strip()
            except Exception:
                return ""

    def _is_noise_line(s: str) -> bool:
        if not s:
            return True
        s2 = s.strip()
        if re.fullmatch(r"[\-\u2500\u2014\u2013\._\s■]+", s2):
            return True
        return False

    rows = _find_rows(table_block)
    if not rows:
        return None

    # conservative: only flatten exact 2-column tables
    ncols = len(rows[0])
    if ncols != 2:
        return None
    if len(rows) < 3:
        return None

    left_cells: list[str] = []
    right_cells: list[str] = []
    for r in rows:
        if not isinstance(r, list) or len(r) < 2:
            continue
        left = _cell_text(r[0])
        right = _cell_text(r[1])
        if not _is_noise_line(left):
            left_cells.append(left)
        if not _is_noise_line(right):
            right_cells.append(right)

    left_combined = "\n".join([c for c in left_cells if c]).strip()
    right_combined = "\n".join([c for c in right_cells if c]).strip()

    # Require both sides to be substantial and mostly CJK to avoid flattening tabular data
    if len(left_combined) < 80 or len(right_combined) < 80:
        return None
    if not (
        _is_mostly_cjk(left_combined, threshold=0.4)
        and _is_mostly_cjk(right_combined, threshold=0.4)
    ):
        return None

    # Return two Para blocks (left then right). Use Str in the inline list so
    # downstream text extraction functions treat them like normal Paras.
    left_para = Para([Str(left_combined)])
    right_para = Para([Str(right_combined)])
    return [left_para, right_para]


def _filter_blocks_recursive(blocks: list[Any], exclude_types: set[str]) -> list[Any]:
    """Recursively filter out blocks whose class names are in exclude_types."""
    filtered: list[Any] = []
    for blk in blocks:
        if blk.__class__.__name__ in exclude_types:
            # Special-case Table: try to flatten simple two-column "newspaper" tables
            if blk.__class__.__name__ == "Table":
                try:
                    flattened = _flatten_table_to_paragraphs(blk)
                    if flattened:
                        filtered.extend(flattened)
                        continue
                except Exception:
                    logging.debug("Table flatten failed", exc_info=True)
            logging.debug(f"Deleting: {blk}")
            continue

        # Handle Div blocks with Attr (id, classes, keyvals).
        # If Div's classes or key-value attributes indicate structural pieces
        # (abstract, author_bio, references, etc.), drop the Div entirely.
        if blk.__class__.__name__ == "Div":
            try:
                # Attr shape: (id, classes, keyvals); inner blocks follow
                attr = blk[0] if len(blk) > 0 else None
                inner = blk[1] if len(blk) > 1 else []
            except Exception:
                # Unexpected shape: keep as-is
                filtered.append(blk)
                continue

            # Extract classes and keyvals safely
            classes = []
            keyvals = []
            if attr:
                try:
                    classes = attr[1] if len(attr) > 1 else []
                    keyvals = attr[2] if len(attr) > 2 else []
                except Exception:
                    classes = []
                    keyvals = []

            # Normalize classes and check for exclusion
            norm_classes = set()
            if classes:
                if isinstance(classes, (list, tuple)):
                    for c in classes:
                        if c:
                            norm_classes.add(str(c).strip().lower().replace("-", "_"))
                elif isinstance(classes, str):
                    norm_classes.add(classes.strip().lower().replace("-", "_"))

            if norm_classes & _DIV_ATTR_EXCLUDE_NORMALIZED:
                logging.debug("Dropping Div by class: %s", norm_classes)
                continue

            # Check key/value attributes for section-like hints (e.g. section=abstract)
            matched = False
            if isinstance(keyvals, list):
                for kv in keyvals:
                    try:
                        k = str(kv[0]).lower()
                        v = kv[1]
                    except Exception:
                        continue
                    if not v:
                        continue
                    if k in {
                        "section",
                        "category",
                        "subtype",
                        "subtypes",
                        "label",
                        "labels",
                        "class",
                    } or (
                        isinstance(k, str)
                        and k.startswith("data-")
                        and k[5:]
                        in {
                            "section",
                            "category",
                            "subtype",
                            "subtypes",
                            "label",
                            "labels",
                        }
                    ):
                        if isinstance(v, str):
                            parts = [
                                p.strip().lower().replace("-", "_")
                                for p in re.split(r"[,/;]+", v)
                                if p.strip()
                            ]
                            if any(p in _DIV_ATTR_EXCLUDE_NORMALIZED for p in parts):
                                matched = True
                                break
                        elif isinstance(v, (list, tuple)):
                            for item in v:
                                si = str(item).strip().lower().replace("-", "_")
                                if si in _DIV_ATTR_EXCLUDE_NORMALIZED:
                                    matched = True
                                    break
                    if matched:
                        break
            if matched:
                logging.debug(
                    "Dropping Div by keyval attr (attr=%s), classes=%s",
                    getattr(attr, 0, None),
                    norm_classes,
                )
                continue

            # Not excluded: recurse into children and re-emit Div only if children remain
            new_inner = _filter_blocks_recursive(inner, exclude_types) if inner else []
            if new_inner:
                filtered.append(Div((attr, new_inner)))
            # If new_inner is empty, drop the Div entirely
            continue

        # Handle nested block containers
        if isinstance(blk, BlockQuote):
            filtered.append(BlockQuote(_filter_blocks_recursive(blk[0], exclude_types)))
        elif isinstance(blk, BulletList):
            # BulletList([ [Block], [Block], ... ])
            new_items = [
                _filter_blocks_recursive(item, exclude_types) for item in blk[0]
            ]
            filtered.append(BulletList(new_items))
        elif isinstance(blk, OrderedList):
            # OrderedList((attrs, [ [Block], ... ]))
            attrs, items = blk
            new_items = [
                _filter_blocks_recursive(item, exclude_types) for item in items
            ]
            filtered.append(OrderedList((attrs, new_items)))
        else:
            filtered.append(blk)
    return filtered


def parse_markdown_blocks(
    text: str, exclude_types: set[str] | None = None
) -> list[Any]:
    """
    Returns Pandoc AST blocks from markdown text, optionally excluding certain block types.

    :param text: Markdown text to parse.
    :param exclude_types: Set of Pandoc block class names to drop entirely.
                          Defaults to {"Table"} to avoid huge table nodes.
    """
    if exclude_types is None:
        exclude_types = {"Table", "Figure"}

    doc_ast = pandoc.read(
        _remove_html_br_tags(text), format="markdown", options=["--wrap=none"]
    )
    blocks = doc_ast[1]
    if exclude_types:
        blocks = _filter_blocks_recursive(blocks, exclude_types)
    return blocks


def parse_markdown_blocks_cached(
    file_path: Path,
    text: str,
    exclude_types: set[str] | None = None,
    force: bool = False,
) -> tuple[list[Any], str]:
    """
    Read pandoc AST for a given file, caching the AST to disk keyed by (path, mtime_ns, size).
    This avoids repeated expensive pandoc subprocess calls for files that do not change.
    """
    # sanitize HTML <br> variants before parsing / hashing fallback
    sanitized_text = _remove_html_br_tags(text)
    if exclude_types is None:
        exclude_types = {"Table", "Figure"}

    # key: resolved path + mtime_ns + size -> hash
    try:
        st = file_path.stat()
        key = f"{file_path.resolve().as_posix()}|{st.st_mtime_ns}|{st.st_size}"
    except Exception:
        # fallback to hashing the sanitized content
        key = xxhash.xxh64(sanitized_text.encode("utf-8")).hexdigest()

    cache_path = CACHE_DIR / (xxhash.xxh64(key.encode("utf-8")).hexdigest() + ".pkl")

    # If caller requested a forced reparse, skip using the existing cache even if present.
    if not force and cache_path.exists():
        try:
            with open(cache_path, "rb") as cf:
                doc_ast = pickle.load(cf)
            blocks = doc_ast[1]
            if exclude_types:
                blocks = _filter_blocks_recursive(blocks, exclude_types)

            # Build canonical plaintext from AST blocks (plain text per block joined by double newlines).
            plain_blocks = [_extract_text_from_blocks([blk]) for blk in blocks]
            canonical_plain = "\n\n".join(pb for pb in plain_blocks if pb)
            canonical_plain = remove_cjk_whitespace(canonical_plain)
            return blocks, canonical_plain
        except Exception:
            logging.debug("Pandoc cache read failed, reparsing", exc_info=True)

    # real parse (expensive)
    doc_ast = pandoc.read(sanitized_text, format="markdown", options=["--wrap=none"])
    blocks = doc_ast[1]
    if exclude_types:
        blocks = _filter_blocks_recursive(blocks, exclude_types)

    # safe atomic write of cache (best-effort)
    try:
        tmp_path = cache_path.with_suffix(".pkl.tmp")
        with open(tmp_path, "wb") as cf:
            pickle.dump(doc_ast, cf, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, cache_path)
    except Exception:
        logging.debug("Pandoc cache write failed", exc_info=True)

    # Return the parsed blocks and the canonical plain text constructed from the AST.
    plain_blocks = [_extract_text_from_blocks([blk]) for blk in blocks]
    canonical_plain = "\n\n".join(pb for pb in plain_blocks if pb)
    canonical_plain = remove_cjk_whitespace(canonical_plain)
    return blocks, canonical_plain


def extract_section_nodes(
    blocks: list[Any],
    title: str,
    subtitle: str | None = None,
    sanitized_text: str | None = None,
    skip_classification: bool | None = None,
) -> list[SectionNode]:
    """Extract headings from blocks into SectionNode list (no hierarchy assigned)."""
    logging.debug(
        f"Extracting section nodes: total blocks={len(blocks)}, title='{title}'"
    )
    nodes: list[SectionNode] = []
    # If not explicitly provided, autodetect per-file: if any Header has attributes,
    # treat the file as "sections labeled" and skip all rule-based classification.
    if skip_classification is None:
        skip_classification = _blocks_have_header_attrs(blocks)
    # Build canonical plaintext from AST blocks and compute deterministic block offsets.
    plain_blocks = [_extract_text_from_blocks([blk]) for blk in blocks]
    sep = "\n\n"

    # Normalize each block's extracted text so offsets are computed in the same
    # normalized space that we use elsewhere (remove_cjk_whitespace).
    plain_blocks_norm = [remove_cjk_whitespace(pb) if pb else "" for pb in plain_blocks]

    # Construct canonical plain text from normalized block texts.
    canonical_plain = sep.join(pb for pb in plain_blocks_norm if pb)

    # Prefer the AST-derived canonical plaintext for slicing/offsets (normalized).
    if sanitized_text is None or sanitized_text != canonical_plain:
        if sanitized_text is not None and sanitized_text != canonical_plain:
            logging.debug(
                "AST-derived canonical plain text differs from provided sanitized_text; using AST canonical form for offsets."
            )
        sanitized_text = canonical_plain

    # Deterministic offsets aligned to canonical_plain: only consider blocks that produced non-empty normalized text.
    char_offsets: list[int | None] = [None] * len(plain_blocks_norm)
    cursor = 0
    included_indices = [i for i, pb in enumerate(plain_blocks_norm) if pb]
    for pos, i in enumerate(included_indices):
        pb = plain_blocks_norm[i] or ""
        char_offsets[i] = cursor
        cursor += len(pb)
        # add separator between included blocks only
        if pos < len(included_indices) - 1:
            cursor += len(sep)

    def block_offset(idx: int) -> int | None:
        return char_offsets[idx] if 0 <= idx < len(char_offsets) else None

    for idx, block in enumerate(blocks):
        if isinstance(block, Header):
            heading_text_raw = remove_cjk_whitespace(
                _extract_text_from_blocks([block]).strip()
            )
            # Do not treat table/figure captions as document sections (e.g. "表1", "表 2", "図 ①")
            if _TABLE_OR_FIG_HEADING_RE.match(heading_text_raw):
                logging.debug(
                    "Skipping table/figure caption heading: %r at block_idx=%d",
                    heading_text_raw,
                    idx,
                )
                continue

            # Avoid treating very long paragraph-like headers as section headings.
            # If a Header block is long and does not contain a numeric prefix, does not
            # match any ordered rule in the HEAD_MATCH_WINDOW, and fails the concise
            # heading heuristic, then it is more likely a body paragraph incorrectly
            # parsed as a Header; skip it.
            if (
                len(heading_text_raw) > 120
                and not NUMBER_PREFIX_RE.match(heading_text_raw)
                and not any(
                    rx.match(heading_text_raw)
                    or rx.search(heading_text_raw[:HEAD_MATCH_WINDOW])
                    for rx, _ in ORDERED_SECTION_RULES
                )
                and not _looks_like_heading_candidate(heading_text_raw)
            ):
                logging.debug(
                    "Skipping long paragraph-like Header block (likely not a section): %r at block_idx=%d",
                    (heading_text_raw[:200] + "...")
                    if len(heading_text_raw) > 200
                    else heading_text_raw,
                    idx,
                )
                continue

            # Skip headings that are non-informative, duplicates of title/subtitle, or
            # too short at deep levels — but allow very short headings when they
            # match an ordered section rule (e.g. "注" -> notes, "目次" -> toc).
            if (
                not re.search(r"[一-龯ぁ-んァ-ンA-Za-z0-9]", heading_text_raw)
                or heading_text_raw == title
                or (subtitle is not None and heading_text_raw == subtitle)
            ):
                continue

            if len(heading_text_raw) < 3 and block[0] > 2:
                # allow short headings that match our ordered section rules in the head window
                head_slice = heading_text_raw[:HEAD_MATCH_WINDOW]
                if not any(
                    rx.match(heading_text_raw) or rx.search(head_slice)
                    for rx, _ in ORDERED_SECTION_RULES
                ):
                    continue

            # --- New behavior: prefer explicit header attributes / classes when present ---
            explicit_matched: set[str] = set()
            try:
                attr = block[1]
            except Exception:
                attr = None

            if attr:
                try:
                    # Pandoc Attr typically: (id, classes, keyvals)
                    _id = attr[0] if len(attr) > 0 else ""
                    classes = attr[1] if len(attr) > 1 else []
                    keyvals = attr[2] if len(attr) > 2 else []
                except Exception:
                    classes = []
                    keyvals = []

                # collect classes (if any)
                if isinstance(classes, (list, tuple)):
                    for c in classes:
                        if c:
                            explicit_matched.add(str(c).strip())
                elif classes:
                    explicit_matched.add(str(classes).strip())

                # collect key/value attributes that look like section/category/subtype info
                if isinstance(keyvals, list):
                    for kv in keyvals:
                        try:
                            k = str(kv[0]).lower()
                            v = kv[1]
                        except Exception:
                            continue
                        if not v:
                            continue
                        if k in {
                            "section",
                            "category",
                            "subtype",
                            "subtypes",
                            "label",
                            "labels",
                        } or (
                            isinstance(k, str)
                            and k.startswith("data-")
                            and k[5:]
                            in {
                                "section",
                                "category",
                                "subtype",
                                "subtypes",
                                "label",
                                "labels",
                            }
                        ):
                            if isinstance(v, str):
                                parts = [
                                    p.strip()
                                    for p in re.split(r"[,/;]+", v)
                                    if p.strip()
                                ]
                                for p in parts:
                                    explicit_matched.add(p)
                            elif isinstance(v, (list, tuple)):
                                for item in v:
                                    if item:
                                        explicit_matched.add(str(item).strip())

            # If header has explicit attributes, use them as authoritative labels.
            if explicit_matched:
                num_prefix = None
                num_val = None
                m = NUMBER_PREFIX_RE.match(heading_text_raw)
                if m:
                    num_prefix = m.group(0).strip()
                    num_val = _parse_int_from_prefix(num_prefix)

                start_pos = block_offset(idx)
                if start_pos is not None:
                    node = SectionNode(
                        level=block[0],
                        num=num_val,
                        num_prefix=num_prefix,
                        raw_text=heading_text_raw,
                        start_char=start_pos,
                        # Expand subtypes into categories so parents/propagation work
                        category=_expand_with_parents(explicit_matched)
                        if explicit_matched
                        else None,
                        subtypes=explicit_matched if explicit_matched else None,
                        matched_subtypes=explicit_matched if explicit_matched else None,
                        block_idx=idx,
                        block=block,
                    )
                    logging.debug(
                        "  Found heading (explicit attrs): level=%s, num=%s, prefix=%r, subtypes=%s, start_char=%s, raw=%r",
                        node.level,
                        node.num,
                        node.num_prefix,
                        node.matched_subtypes,
                        node.start_char,
                        (node.raw_text[:200] + "...")
                        if len(node.raw_text or "") > 200
                        else node.raw_text,
                    )
                    nodes.append(node)
                continue

            # No explicit attrs: if file-level skip_classification is True, do NOT run any
            # rule-based normalization or spaCy heuristics; emit unlabeled nodes only.
            if skip_classification:
                num_prefix = None
                num_val = None
                m = NUMBER_PREFIX_RE.match(heading_text_raw)
                if m:
                    num_prefix = m.group(0).strip()
                    num_val = _parse_int_from_prefix(num_prefix)

                start_pos = block_offset(idx)
                if start_pos is not None:
                    node = SectionNode(
                        level=block[0],
                        num=num_val,
                        num_prefix=num_prefix,
                        raw_text=heading_text_raw,
                        start_char=start_pos,
                        category=None,
                        subtypes=None,
                        matched_subtypes=None,
                        block_idx=idx,
                        block=block,
                    )
                    logging.debug(
                        "  Found heading (no classification): level=%s, num=%s, prefix=%r, start_char=%s, raw=%r",
                        node.level,
                        node.num,
                        node.num_prefix,
                        node.start_char,
                        (node.raw_text[:200] + "...")
                        if len(node.raw_text or "") > 200
                        else node.raw_text,
                    )
                    nodes.append(node)
                continue

            # Fallback (no explicit attrs, not skipping): use existing rule-based normalization.
            norm, num_prefix, subtypes = _normalize_section_name(heading_text_raw)
            num = None
            if num_prefix:
                num = _parse_int_from_prefix(num_prefix)
            start_pos = block_offset(idx)
            if start_pos is not None:
                node = SectionNode(
                    level=block[0],
                    num=num,
                    num_prefix=num_prefix,
                    raw_text=heading_text_raw,
                    category=norm,
                    subtypes=subtypes,
                    matched_subtypes=subtypes,
                    block_idx=idx,
                    start_char=start_pos,
                    block=block,
                )
                logging.debug(
                    f"  Found heading: level={node.level}, num={node.num}, "
                    f"prefix={node.num_prefix}, category={node.category}, start_char={node.start_char}, "
                    f"raw='{node.raw_text}'"
                )
                nodes.append(node)

        elif isinstance(block, (Para, Plain)):
            # In annotated mode, do not synthesize headings from paragraph-like blocks
            if skip_classification:
                continue
            heading_text_raw = remove_cjk_whitespace(
                _extract_text_from_blocks([block]).strip()
            )
            if not heading_text_raw:
                continue

            # Skip obvious table/figure captions or duplicates of title/subtitle
            if _TABLE_OR_FIG_HEADING_RE.match(heading_text_raw):
                logging.debug(
                    "Skipping table/figure caption para: %r at block_idx=%d",
                    heading_text_raw,
                    idx,
                )
                continue
            if heading_text_raw == title or (
                subtitle is not None and heading_text_raw == subtitle
            ):
                continue

            # Consider a Para/Plain as a heading only when it matches an ordered rule
            # or begins with a numeric prefix. Also allow the heuristic _looks_like_heading_candidate
            # but only for relatively short paragraph fragments to avoid promoting long
            # multi-sentence paragraphs (e.g. English abstract sentences) into headers.
            head_slice = heading_text_raw[:HEAD_MATCH_WINDOW]
            is_section_like = False
            if any(
                rx.match(heading_text_raw) or rx.search(head_slice)
                for rx, _ in ORDERED_SECTION_RULES
            ):
                is_section_like = True
            elif NUMBER_PREFIX_RE.match(heading_text_raw):
                is_section_like = True
            else:
                # only apply heuristic for relatively short paragraph headings (<= 60 chars)
                if len(heading_text_raw) <= 60 and _looks_like_heading_candidate(
                    heading_text_raw
                ):
                    is_section_like = True

            if not is_section_like:
                continue

            # If skipping classification for this file, do not invoke rule-based mapping
            if skip_classification:
                num_prefix = None
                num = None
                m2 = NUMBER_PREFIX_RE.match(heading_text_raw)
                if m2:
                    num_prefix = m2.group(0).strip()
                    num = _parse_int_from_prefix(num_prefix)
                start_pos = block_offset(idx)
                if start_pos is not None:
                    node = SectionNode(
                        level=2,
                        num=num,
                        num_prefix=num_prefix,
                        raw_text=heading_text_raw,
                        category=None,
                        subtypes=None,
                        matched_subtypes=None,
                        block_idx=idx,
                        start_char=start_pos,
                        block=block,
                    )
                    logging.debug(
                        "  Found para-heading (no classification): start=%d prefix=%r raw=%r",
                        start_pos,
                        num_prefix,
                        heading_text_raw,
                    )
                    nodes.append(node)
                continue

            # Otherwise, original normalization path
            norm, num_prefix, subtypes = _normalize_section_name(heading_text_raw)
            num = None
            if num_prefix:
                num = _parse_int_from_prefix(num_prefix)
            start_pos = block_offset(idx)
            if start_pos is not None:
                node = SectionNode(
                    level=2,
                    num=num,
                    num_prefix=num_prefix,
                    raw_text=heading_text_raw,
                    category=norm,
                    subtypes=subtypes,
                    matched_subtypes=subtypes,
                    block_idx=idx,
                    start_char=start_pos,
                    block=block,
                )
                logging.debug(
                    "  Found para-heading (promoted to level=2): start=%d prefix=%r raw=%r",
                    start_pos,
                    num_prefix,
                    heading_text_raw,
                )
                nodes.append(node)

    # --- Repair OCR'd numeric prefixes and re-parse numeric values ---
    for n in nodes:
        if n.num_prefix:
            try:
                repaired = _repair_numeric_prefix_ocr(n.num_prefix)
                if repaired and repaired != n.num_prefix:
                    logging.debug(
                        "Repairing numeric prefix %r -> %r for header %r",
                        n.num_prefix,
                        repaired,
                        (n.raw_text or "")[:120],
                    )
                    n.num_prefix = repaired
                parsed = _parse_int_from_prefix(n.num_prefix)
                if parsed is not None:
                    n.num = parsed
            except Exception:
                # be conservative: leave original values on failure
                logging.debug(
                    "Numeric prefix repair/parsing failed for %r",
                    n.num_prefix,
                    exc_info=True,
                )

    # Exclude detected title/subtitle from the content-level scan only when we may rebase levels.
    if not skip_classification:
        content_nodes = [
            n
            for n in nodes
            if n.raw_text
            and n.raw_text != (title or "")
            and n.raw_text != (subtitle or "")
            and not _looks_like_author_line(n.raw_text)
            and not _TABLE_OR_FIG_HEADING_RE.match(n.raw_text or "")
        ]
        if content_nodes:
            try:
                min_content_level = min(
                    (n.level for n in content_nodes if n.level), default=None
                )
                if min_content_level is not None and min_content_level != 2:
                    shift = min_content_level - 2
                    logging.debug(
                        "Rebasing section levels: min_content_level=%d, shift=%d",
                        min_content_level,
                        shift,
                    )
                    for n in nodes:
                        if n.level:
                            # shifting works in both directions: positive shift lowers levels,
                            # negative shift raises them (so H1 -> H2 when necessary)
                            n.level = max(1, (n.level or 1) - shift)
            except Exception:
                logging.debug("Failed to rebase header levels", exc_info=True)

    # Post-check: log any header nodes where the expected header_text does not appear
    # in the immediate sanitized_text window. This helps spot files that need better heuristics.
    if sanitized_text is not None:
        doc_len = len(sanitized_text)
        for n in nodes:
            if not getattr(n, "block", None) or not isinstance(n.block, Header):
                continue
            try:
                expected = remove_cjk_whitespace(
                    _extract_text_from_blocks([n.block]).strip()
                )
                if not expected:
                    continue
                start = n.start_char if n.start_char is not None else 0
                # small context window around start to tolerate small alignment differences
                ws = max(0, start - HEAD_MATCH_WINDOW)
                we = min(doc_len, start + len(expected) + HEAD_MATCH_WINDOW)
                window = sanitized_text[ws:we]
                if expected not in window:
                    logging.debug(
                        "Header alignment mismatch: raw=%r start=%d window_preview=%r expected_head=%r",
                        (n.raw_text[:80] + "...")
                        if len(n.raw_text or "") > 80
                        else n.raw_text,
                        start,
                        (window[:200] + "...") if len(window) > 200 else window,
                        (expected[:200] + "...") if len(expected) > 200 else expected,
                    )
            except Exception:
                logging.debug(
                    "Header alignment check failed for node at start=%r",
                    getattr(n, "start_char", None),
                    exc_info=True,
                )
    logging.debug(f"Total real section nodes extracted: {len(nodes)}")
    return nodes


def merge_synthetic_nodes(real: list[SectionNode], text: str) -> list[SectionNode]:
    """Find missing numbered sections in text and return combined node list.

    Improvements vs. previous implementation:
      - Determine "top-level" numeric anchors by prefix style (e.g. '1.' or '1.2' are
        considered explicit anchors; '(1)' and '①' are considered subordinate).
      - Deduplicate nodes (prefer real/blocked nodes over synthetic ones when start_char/raw_text collide).
      - Preserve conservative ordering checks to avoid inserting stray numeric matches.
    """
    # Determine explicit top-level numbers from real headings (do not rely on pandoc levels)
    top_level_nums = sorted(
        n.num
        for n in real
        if n.num is not None and _is_explicit_section_prefix(n.num_prefix)
    )
    logging.debug(
        "Merging synthetic nodes: real_count=%d, explicit_top_level_nums=%s",
        len(real),
        top_level_nums,
    )
    if not top_level_nums:
        # Nothing to anchor synthetic nodes to; return real nodes (deduped)
        # Deduplicate real nodes by (start_char, raw_text)
        seen = {}
        out = []
        for n in sorted(
            real, key=lambda x: (x.start_char, 0 if not x.synthetic else 1)
        ):
            key = (n.start_char, remove_cjk_whitespace((n.raw_text or "").strip()))
            if key in seen:
                continue
            seen[key] = n
            out.append(n)
        return out

    min_num, max_num = min(top_level_nums), max(top_level_nums)
    synthetic_nodes: list[SectionNode] = []
    # Precompute earliest start positions for real nodes by numeric label.
    num_min_start: dict[int, int] = {}
    for n in real:
        if n.num is None:
            continue
        if n.num not in num_min_start or n.start_char < num_min_start[n.num]:
            num_min_start[n.num] = n.start_char

    patterns = _PARA_NUM_PREFIX_PATTERNS
    seen_positions: set[int] = set()

    # scan line-by-line to avoid overlapping matches and to get reliable offsets
    for line_match in re.finditer(r"^(.*)$", text, flags=re.M):
        line = line_match.group(1)
        line_start = line_match.start()
        for pat in patterns:
            mm = pat.match(line)
            if not mm:
                continue
            prefix = mm.group(1)
            body = mm.group(2).strip() if mm.lastindex and mm.lastindex >= 2 else ""
            num = _parse_int_from_prefix(prefix)
            if num is None:
                continue
            # Do not duplicate an existing explicit top-level section
            if num in num_min_start and _is_explicit_section_prefix(prefix):
                # there is already a real explicit heading with this number -> skip
                continue
            # be conservative: only create synthetic nodes for numbers within observed range
            if num < min_num or num > max_num:
                continue
            # compute absolute start position for the body text robustly from match groups
            if mm.lastindex and mm.lastindex >= 2:
                start_pos = line_start + mm.start(2)
            else:
                body_idx = line.find(body)
                start_pos = (
                    line_start + body_idx
                    if body_idx >= 0
                    else text.find(body, line_start)
                )
            # Ensure the prefix occurs near line start (avoid matching numbers later in sentences)
            try:
                if mm.end(1) > 8:
                    logging.debug(
                        "Skipping numeric prefix because it is not near line start: prefix=%r, line_start=%d",
                        prefix,
                        line_start,
                    )
                    continue
            except Exception:
                continue

            # Ordering guard: reject if a higher-numbered explicit heading occurs earlier
            higher_starts = [s for k, s in num_min_start.items() if k > num]
            if higher_starts and start_pos > min(higher_starts):
                logging.debug(
                    "Skipping candidate num=%r at pos=%d because a higher-numbered section occurs earlier at %d",
                    num,
                    start_pos,
                    min(higher_starts),
                )
                continue

            if start_pos is None or start_pos < 0 or start_pos in seen_positions:
                continue
            seen_positions.add(start_pos)
            heading_candidate = remove_cjk_whitespace(body)
            if not any(rx.search(heading_candidate) for rx, _ in ORDERED_SECTION_RULES):
                if not _looks_like_heading_candidate(heading_candidate):
                    continue
            norm, num_prefix, subtypes = _normalize_section_name(heading_candidate)
            if not num_prefix:
                num_prefix = f"{num}."
            syn_node = SectionNode(
                level=2,  # synthetic anchors are assumed section-level (canonical)
                num=num,
                num_prefix=num_prefix,
                raw_text=heading_candidate,
                category=norm,
                subtypes=subtypes,
                matched_subtypes=subtypes,
                block_idx=-1,
                start_char=start_pos,
                synthetic=True,
            )
            logging.debug(
                "  Added synthetic node: num=%d, prefix=%r, category=%s, start_char=%d, raw=%r",
                syn_node.num,
                syn_node.num_prefix,
                syn_node.category,
                syn_node.start_char,
                syn_node.raw_text,
            )
            synthetic_nodes.append(syn_node)
            break  # stop trying other patterns for this line

    combined = real + synthetic_nodes

    # Deduplicate conservative: prefer real/block nodes over synthetic when (start_char, raw_text) collide.
    seen = {}
    dedup_nodes: list[SectionNode] = []
    for n in sorted(
        combined, key=lambda x: (x.start_char, 0 if not x.synthetic else 1)
    ):
        key = (n.start_char, remove_cjk_whitespace((n.raw_text or "").strip()))
        if key in seen:
            # if existing is synthetic but current is real, replace
            existing = seen[key]
            if existing.synthetic and not n.synthetic:
                seen[key] = n
                # update last appended item
                for i in range(len(dedup_nodes) - 1, -1, -1):
                    if (
                        dedup_nodes[i].start_char,
                        remove_cjk_whitespace((dedup_nodes[i].raw_text or "").strip()),
                    ) == key:
                        dedup_nodes[i] = n
                        break
            continue
        seen[key] = n
        dedup_nodes.append(n)

    logging.debug(
        "Total nodes after merge: %d (real=%d synthesized=%d)",
        len(dedup_nodes),
        len(real),
        len(synthetic_nodes),
    )
    return sorted(dedup_nodes, key=lambda n: n.start_char)


def build_section_tree(nodes: list[SectionNode], doc_len: int) -> SectionNode:
    """Builds a hierarchy of sections from flat node list."""
    logging.debug(f"Building section tree from {len(nodes)} nodes, doc_len={doc_len}")

    root = SectionNode(
        level=0,
        num=None,
        num_prefix=None,
        raw_text="ROOT",
        category=None,
        block_idx=-1,
        start_char=0,
        end_char=doc_len,
    )
    stack = [root]
    for node in nodes:
        while stack and stack[-1].level >= node.level:
            stack.pop()
        stack[-1].children.append(node)
        stack.append(node)
    assign_end_chars(root, doc_len)
    collapse_same_category(root)

    def _dump(node: SectionNode, indent=0):
        logging.debug(
            f"{' ' * indent}Node(level={node.level}, num={node.num}, "
            f"prefix={node.num_prefix}, category={node.category}, start={node.start_char}, end={node.end_char}, "
            f"synthetic={node.synthetic})"
        )
        for ch in node.children:
            _dump(ch, indent + 2)

    _dump(root)
    return root


def assign_end_chars(node: SectionNode, doc_len: int) -> None:
    node.children.sort(key=lambda c: c.start_char)
    for i, child in enumerate(node.children):
        child.end_char = (
            node.children[i + 1].start_char
            if i + 1 < len(node.children)
            else node.end_char or doc_len
        )
        assign_end_chars(child, doc_len)


def collapse_same_category(
    node: SectionNode, parent_cat: set[str] | None = None
) -> None:
    if node.category == parent_cat:
        node.category = None
    for ch in node.children:
        collapse_same_category(ch, node.category or parent_cat)


def _count_numeric_components(prefix: str) -> int:
    """Count simple numeric components in a prefix.

    Examples:
    >>> _count_numeric_components("1.2.3")
    3
    >>> _count_numeric_components("①")
    1
    """
    if not prefix:
        return 0
    groups = NUMERIC_COMPONENTS_RE.findall(prefix)
    return len(groups)


def _prefix_style(prefix: str | None) -> str:
    """Return a short code describing the numeric-prefix style.

    Recognised styles (ordered by precedence): parenthesis, circled, dot, roman,
    kanji, plain, none.

    Examples:
    >>> _prefix_style("(1)")
    'parenthesis'
    >>> _prefix_style("1.")
    'dot'
    >>> _prefix_style("①")
    'circled'
    """
    if not prefix:
        return "none"
    p = prefix.strip()
    # parenthesised: "(1)" or "（1）"
    if re.match(r"^[（(].+[)）]$", p):
        return "parenthesis"
    # circled: ①..⑳
    if _CIRCLED_RE.search(p):
        return "circled"
    # dot/hyphen separated: 1. 1.2 1-2 1． (fullwidth dot)
    if re.search(r"[.\-．－]", p):
        return "dot"
    # roman (ASCII or unicode)
    if re.search(r"[IVXLCDMivxlcdmⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫ]", p):
        return "roman"
    # kanji numerals
    if _KANJI_NUM_RE.search(p):
        return "kanji"
    # plain numeric like "1"
    if re.search(r"[\d０-９]", p):
        return "plain"
    return "other"


def _is_explicit_section_prefix(prefix: str | None) -> bool:
    """Return True if the numeric prefix is an explicit section anchor (top-level).

    Heuristics:
      - Multi-component prefixes (1.2, 1-2) or dot-terminated forms ("1.", "2.") are explicit.
      - Roman numerals are treated as explicit.
      - Parenthesised / circled / simple kanji-only forms are considered subordinate by default.

    Examples:
    >>> _is_explicit_section_prefix("1.")
    True
    >>> _is_explicit_section_prefix("(1)")
    False
    >>> _is_explicit_section_prefix("1.2")
    True
    >>> _is_explicit_section_prefix("①")
    False
    """
    if not prefix:
        return False
    style = _prefix_style(prefix)
    if style in {"dot", "roman"}:
        return True
    if style in {"parenthesis", "circled"}:
        return False
    # multi-component numeric tokens -> explicit
    if _count_numeric_components(prefix) > 1:
        return True
    # plain ascii digit (single): treat as explicit (e.g. "5")
    if style == "plain":
        return True
    # kanji: be conservative (treat as subordinate unless it contains a separator)
    if style == "kanji":
        return bool(re.search(r"[.\-．－]", prefix))
    return False


def _kanji_to_int(s: str) -> int:
    """Parse simple kanji numerals (up to thousands) into an integer."""
    small = {
        "一": 1,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    mult = {"十": 10, "百": 100, "千": 1000}
    val = 0
    tmp = 0
    for ch in s:
        if ch in small:
            tmp += small[ch]
        elif ch in mult:
            if tmp == 0:
                tmp = 1
            tmp *= mult[ch]
            val += tmp
            tmp = 0
    val += tmp
    return val


def _parse_int_from_prefix(prefix: str) -> int | None:
    """Try to extract an integer from a numeric prefix that may be ASCII, fullwidth,
    circled, roman (single unicode chars), or simple kanji.

    Returns None if no sensible integer found.
    """
    if not prefix:
        return None
    # 1) arabic digits (incl fullwidth)
    m = _ARABIC_DIGITS_RE.search(prefix)
    if m:
        digits = m.group(0)
        tr = {ord("０") + i: ord("0") + i for i in range(10)}
        digits = digits.translate(tr)
        try:
            return int(digits)
        except Exception:
            pass

    # 2) circled numbers ①..⑳ (U+2460..U+2473)
    circ = _CIRCLED_RE.search(prefix)
    if circ:
        return ord(circ.group(0)) - 0x2460 + 1

    # 3) ASCII Roman numerals (I, II, III, IV, V, ...). Convert generically.
    ascii_rom = re.search(r"[IVXLCDMivxlcdm]+", prefix)
    if ascii_rom:
        s = ascii_rom.group(0).upper()

        def _roman_to_int(s_rom: str) -> int:
            vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
            total = 0
            prev = 0
            for ch in reversed(s_rom):
                v = vals.get(ch, 0)
                if v < prev:
                    total -= v
                else:
                    total += v
                    prev = v
            return total

        try:
            val = _roman_to_int(s)
            # guard against nonsense outputs (e.g. empty)
            if val > 0:
                return val
        except Exception:
            pass

    # 4) roman numerals (common single unicode chars)
    ROMAN_MAP = {
        "Ⅰ": 1,
        "Ⅱ": 2,
        "Ⅲ": 3,
        "Ⅳ": 4,
        "Ⅴ": 5,
        "Ⅵ": 6,
        "Ⅶ": 7,
        "Ⅷ": 8,
        "Ⅸ": 9,
        "Ⅹ": 10,
        "Ⅺ": 11,
        "Ⅻ": 12,
    }
    rom = _ROMAN_RE.search(prefix)
    if rom:
        key = rom.group(0)
        if key in ROMAN_MAP:
            return ROMAN_MAP[key]

    # 5) kanji numerals
    kanji = _KANJI_NUM_RE.search(prefix)
    if kanji:
        return _kanji_to_int(kanji.group(0))
    return None


def _repair_numeric_prefix_ocr(prefix: str | None) -> str | None:
    """
    Conservative OCR repair for numeric prefixes.

    - Fix common OCR confusions for Roman numerals where 'l'/'L' was read
      instead of 'I', and a few other minor problems.
    - Return a repaired prefix string or the original when no change is made.
    """
    if not prefix:
        return prefix
    s = prefix.strip()
    # Work on the leading token (the numeric prefix token)
    m = re.match(r"^([^\s]+)", s)
    token = m.group(1) if m else s

    # Roman-like repairs: uppercase and replace letter-L (often OCRed) to I
    if re.search(r"[IVXLCDMivxlcdmL]", token):
        repaired_token = token.upper().replace("L", "I")
        # Preserve any trailing punctuation/spaces in the original prefix
        return s.replace(token, repaired_token, 1)

    # Digit-like repairs: O/o mistaken for zero when digits present
    if re.search(r"[0-9０-９]", token) and re.search(r"[Oo]", token):
        repaired_token = token.replace("O", "0").replace("o", "0")
        return s.replace(token, repaired_token, 1)

    return s


def _looks_like_author_line(s: str) -> bool:
    """
    Heuristic to detect author/affiliation lines so they are not used as 'content' headers.

    Returns True for short lines that contain affiliation keywords or typical affiliation patterns.
    """
    if not s:
        return False
    # common Japanese/English affiliation keywords
    if re.search(
        r"(大学|教授|准教授|助教|名誉|学部|研究科|所属|department|university|professor|Dr\.?)",
        s,
        flags=re.IGNORECASE,
    ):
        return True
    # parenthesised affiliation snippets like "(新潟大学名誉教授)" or "(University of X)"
    if re.search(
        r"[（(].{1,80}(大学|University|Dept|Faculty|研究科|学部|所属).{0,80}[)）]",
        s,
        flags=re.IGNORECASE,
    ):
        return True
    # short lines with parentheses (likely name + affiliation)
    if len(s.split()) <= 6 and ("(" in s or "（" in s):
        return True
    return False


def _promote_numeric_section_prefixes(
    nodes: list[SectionNode], base_level: int = 2, min_candidates: int = 3
) -> None:
    """
    Promote numeric-prefixed nodes (roman/dot/plain) to `base_level` when there is
    clear evidence they form top-level section anchors.

    Heuristics:
      - Consider only nodes with a numeric prefix and not author/table captions.
      - Candidate prefix styles: 'roman', 'dot', 'plain'.
      - Require at least `min_candidates` candidates to avoid noisy promotion.
      - Require numeric variety (>=2 distinct parsed ints) or monotonic increase among parsed ints.
    Operates in-place on `nodes`.
    """
    if not nodes:
        return

    # choose candidates conservatively (prefer real/block nodes)
    candidates = [
        n
        for n in nodes
        if n.num_prefix
        and not _looks_like_author_line(n.raw_text or "")
        and not _TABLE_OR_FIG_HEADING_RE.match(n.raw_text or "")
        and _prefix_style(n.num_prefix) in {"roman", "dot", "plain"}
        and not getattr(n, "synthetic", False)
    ]
    if len(candidates) < min_candidates:
        return

    # sort by document order
    candidates = sorted(candidates, key=lambda n: n.start_char or 0)

    # collect parsed integer values for robustness checks
    parsed_ints = [n.num for n in candidates if isinstance(n.num, int)]
    # require at least two distinct parsed ints (unless many candidates exist)
    if parsed_ints and len(set(parsed_ints)) < 2 and len(candidates) < 5:
        return

    # monotonicity check when we have ints: require at least a majority of pairs to be increasing
    if parsed_ints:
        inc = sum(1 for a, b in zip(parsed_ints, parsed_ints[1:]) if b > a)
        if inc < max(1, len(parsed_ints) // 2):
            return

    # Passed checks: promote these candidates to the canonical base level
    for n in candidates:
        try:
            if (n.level or 0) != base_level:
                logging.debug(
                    "Promoting numeric-prefixed node to base level=%d: start=%s prefix=%r raw=%r",
                    base_level,
                    str(n.start_char),
                    n.num_prefix,
                    (n.raw_text or "")[:120],
                )
                n.level = base_level
        except Exception:
            logging.debug(
                "Failed to promote node at %r",
                getattr(n, "start_char", None),
                exc_info=True,
            )


def _assign_enumerated_hypotheses(nodes: list[SectionNode]) -> None:
    """
    Assign hierarchical numeric prefixes to inline enumerated hypothesis-like headings.

    Example: after a parent with num_prefix "3.2." this will convert
    a sibling raw_text "仮説 1: ..." into num_prefix "3.2.1." and set node.num to the parent's top-level number.
    Operates in-place on the nodes list (which must be sorted by start_char).
    """
    if not nodes:
        return
    enum_re = _ENUM_HYPOTHESIS_RE
    next_seq: dict[str, int] = {}
    for i, node in enumerate(nodes):
        if node.num is not None or not node.raw_text:
            continue
        m = enum_re.match(node.raw_text)
        if not m:
            continue
        # Find nearest preceding node that has an explicit numeric prefix
        base_idx = None
        for j in range(i - 1, -1, -1):
            if nodes[j].num_prefix:
                base_idx = j
                break
        if base_idx is None:
            continue
        base_node = nodes[base_idx]
        base_prefix = base_node.num_prefix or ""
        if not base_prefix.endswith("."):
            base_prefix = base_prefix + "."
        key = base_prefix
        seq = next_seq.get(key, 1)
        # Assign a child numbering (3.2 -> 3.2.1, 3.2.2, ...)
        node.num = base_node.num
        node.num_prefix = f"{base_prefix}{seq}."
        # promote/demote level so node becomes a child of base_node when tree is built
        node.level = (base_node.level or 0) + 1
        next_seq[key] = seq + 1


def _normalize_numeric_sequences(nodes: list[SectionNode]) -> None:
    """Assign stable, monotonic section levels for numeric heading runs.

    Algorithm summary:
      - Use a canonical base level (2) for section runs.
      - Multi-component prefixes (e.g. "1.2.3") map to base + (components-1).
      - Single-component prefixes choose a depth by scoring candidate depths that
        balance monotonicity, prefix style (parenthesised/circled bias deeper,
        dotted/multi-component bias shallower) and closeness to the original pandoc level.
      - If no candidate satisfies monotonicity, the node is attached as a new sublevel
        (prev_level + 1) to preserve a continuous numbering run.
    Operates in-place; nodes must be sorted by start_char.
    """
    if not nodes:
        return
    # ensure document order
    nodes.sort(key=lambda n: n.start_char)
    BASE = 2
    MAX_DEPTH = 8
    last_num_at_level: dict[int, int] = {}
    prev_assigned_level = BASE

    for node in nodes:
        if not node.num_prefix or not NUMERIC_COMPONENTS_RE.search(node.num_prefix):
            continue
        prefix = node.num_prefix or ""
        # extract numeric components (as ints) when available
        raw_tokens = NUMERIC_COMPONENTS_RE.findall(prefix)
        comp_vals = [
            (_parse_int_from_prefix(tok) if tok else None) for tok in raw_tokens
        ]
        comp_vals = [v for v in comp_vals if v is not None]
        comps_count = len(comp_vals)

        style = _prefix_style(prefix)

        # If multi-component, depth is explicit: base + (components - 1)
        if comps_count > 1:
            assigned_level = BASE + (comps_count - 1)
            num_value = comp_vals[-1]
        else:
            # single-component case
            num_value = (
                node.num
                if node.num is not None
                else (comp_vals[0] if comp_vals else None)
            )

            # Score candidate depths and choose best
            best_score = float("inf")
            best_depth: int | None = None
            for d in range(BASE, BASE + MAX_DEPTH):
                last = last_num_at_level.get(d)
                monotonic = (last is None) or (num_value is None) or (num_value > last)
                # large penalty if monotonicity violated
                score = 0.0
                if not monotonic:
                    score += 1e6
                # bias by prefix style: parenthesis/circled/kanji => prefer deeper
                if style in ("parenthesis", "circled", "kanji"):
                    score += max(0, (d - BASE)) * 50.0
                elif style == "dot":
                    # prefer top-level for 'dot'
                    score += abs(d - BASE) * 8.0
                elif style == "roman":
                    score += abs(d - BASE) * 4.0
                else:  # plain / other
                    score += abs(d - BASE) * 2.0
                # prefer closeness to original pandoc level
                score += abs(d - (node.level or BASE)) * 3.0
                # slight preference to shallower depths to avoid overly-deep promotion
                score += d * 0.01
                if score < best_score:
                    best_score = score
                    best_depth = d

            if best_depth is None:
                assigned_level = max(prev_assigned_level + 1, BASE)
            else:
                # if best candidate had enormous penalty (no monotonic depth found), attach as sublevel
                if best_score >= 1e6:
                    assigned_level = max(prev_assigned_level + 1, BASE)
                else:
                    assigned_level = best_depth

            # Force single-component explicit numeric prefixes to the canonical base level.
            # This ensures headings like "3 結果" are treated as top-level section anchors (level BASE),
            # avoiding accidental nesting beneath the previous top-level when many deeply-nested nodes precede them.
            try:
                if (
                    comps_count == 1
                    and style in ("dot", "plain", "roman")
                    and not getattr(node, "synthetic", False)
                ):
                    assigned_level = BASE
            except Exception:
                # be conservative on any unexpected failure: leave assigned_level as computed
                pass

        if assigned_level < 1:
            assigned_level = 1

        # update last seen number for monotonicity tracking
        if num_value is not None:
            last_num_at_level[assigned_level] = num_value
            # clear any deeper levels (they belong to previous branches)
            for deeper in [
                lv for lv in list(last_num_at_level.keys()) if lv > assigned_level
            ]:
                del last_num_at_level[deeper]

        if node.level != assigned_level:
            logging.debug(
                "Normalizing numeric prefix %r at char %d: level %d -> %d",
                node.num_prefix,
                node.start_char,
                node.level,
                assigned_level,
            )
            node.level = assigned_level

        prev_assigned_level = assigned_level


def _smooth_level_jumps(nodes: list[SectionNode]) -> None:
    """Clamp large increases in header levels so that a node's level never jumps
    by more than +1 relative to the previous node (in document order)."""
    if not nodes:
        return
    for idx in range(1, len(nodes)):
        allowed = nodes[idx - 1].level + 1
        if nodes[idx].level > allowed:
            logging.debug(
                "Smoothing large level jump at char %d: %d -> %d",
                nodes[idx].start_char,
                nodes[idx].level,
                allowed,
            )
            nodes[idx].level = allowed


# --- CJK whitespace filter ---
_CJK_RANGE = r"\u4E00-\u9FFF\u3400-\u4DBF\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F\uFF00-\uFFEF"
# Only collapse same-line spaces between CJK — do not match \n
_CJK_SPACE_RE = re.compile(rf"([{_CJK_RANGE}])[ \t\u3000]+([{_CJK_RANGE}])")

# Module-level compiled regexes and simple pandoc AST cache
CACHE_DIR = Path(".cache/pandoc_ast")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Number prefix extraction (moved out of _normalize_section_name)
NUMBER_PREFIX_RE = re.compile(
    r"""^(
            \d+(?:[.\-．－]\d+)*[.\．]?\s*                      # 1, 1-2, 5.3.1, allow fullwidth dot/hyphen
          | \(\d+(?:[.\-．－]\d+)*\)\s*                         # (1), (3-2) etc. allow fullwidth separators
          | [IVXLCDMivxlcdm]+(?![a-z])[.．]?\s*                # ASCII Roman numerals (I, II, III, IV, ...)
          | [一二三四五六七八九十百千]+[.．]?\s*               # Kanji numerals
          | [①-⑳]\s*                                          # circled nums
          | [ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫ]+\s*                               # fullwidth/Unicode Roman numerals
          | [（(]\s*[一二三四五六七八九\d]+(?:[.\-．－]\d+)*\s*[)）]\s*  # ( 2 ), ( 1-2 )
        )""",
    re.VERBOSE,
)

_YAML_FM_RE = re.compile(
    r"^\s*(?P<delim>---|\+\+\+)\s*\n(?P<body>.*?)(?:\n(?P=delim)\s*|\n\.\.\.\s*)",
    re.S,
)


def _parse_yaml_front_matter(text: str) -> dict[str, Any] | None:
    """
    Parse YAML front matter from the beginning of `text` and return a mapping or None.
    """
    if not text:
        return None
    m = _YAML_FM_RE.match(text)
    if not m:
        return None
    body = m.group("body")
    try:
        parsed = yaml.safe_load(body)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        # light-weight fallback parser for tiny lists/values (keeps function robust)
        data: dict[str, Any] = {}
        cur_key: str | None = None
        for line in body.splitlines():
            if not line.strip():
                continue
            # list item
            if re.match(r"^\s*-\s+", line):
                if cur_key:
                    item = line.split("-", 1)[1].strip().strip("\"'")
                    data.setdefault(cur_key, []).append(item)
                continue
            m2 = re.match(r"^\s*([^:]+):\s*(.*)$", line)
            if not m2:
                continue
            key = m2.group(1).strip()
            val = m2.group(2).strip()
            cur_key = key
            if val == "":
                data[key] = []
            else:
                if (val.startswith('"') and val.endswith('"')) or (
                    val.startswith("'") and val.endswith("'")
                ):
                    val = val[1:-1]
                data[key] = val
        return data or None


def _split_keywords(val: Any) -> list[str]:
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    s = str(val).strip()
    if not s:
        return []
    import re

    return [t.strip() for t in re.split(r"[、,/／;；\|\s]+", s) if t.strip()]


def normalize_genre_meta(meta: dict[str, Any]) -> dict[str, Any]:
    corpus = meta.get("corpus")
    subject_area = meta.get("subject_area")
    legacy = meta.get("genre")

    if not corpus:
        if isinstance(legacy, list) and legacy:
            corpus = legacy[0]
            subject_area = subject_area or (legacy[1] if len(legacy) > 1 else None)
        elif isinstance(legacy, str) and legacy.strip():
            corpus = legacy.strip()

    kw = _split_keywords(meta.get("keywords"))
    genre1 = str(corpus).strip() if corpus else None
    genre2 = str(subject_area).strip() if subject_area else None
    genre3 = kw[0] if kw else None
    genre_path = " / ".join([s for s in [genre1, genre2] if s])

    out = dict(meta)
    out["genre"] = genre1
    out["genre1"] = genre1
    out["genre2"] = genre2
    out["keywords"] = kw or None
    out["genre3"] = genre3
    out["genre_path"] = genre_path if genre_path else None
    out["ジャンル"] = genre1
    return out


def _yaml_meta_to_info(yaml_meta: dict[str, Any], path: Path) -> dict[str, Any]:
    """
    Convert YAML front matter into canonical info, preserving corpus/subject_area/keywords.
    - genre is a single string equal to corpus
    - genre1/genre2/genre3 derived from corpus/subject_area/keywords[0]
    - journal_name, volume, number, permission kept when present
    """

    def _to_int(x: Any) -> int | None:
        try:
            return int(x) if x is not None and str(x).strip().isdigit() else None
        except Exception:
            return None

    out: dict[str, Any] = {}
    out["basename"] = str(yaml_meta.get("basename") or path.stem)

    title = (
        yaml_meta.get("title")
        or yaml_meta.get("journal_title")
        or yaml_meta.get("journal_name")
        or out["basename"]
    )
    out["title"] = str(title)

    # Keep both journal_title and journal_name for compatibility
    out["journal_name"] = str(yaml_meta.get("journal_name") or "")
    out["journal_title"] = str(
        yaml_meta.get("journal_title") or yaml_meta.get("journal_name") or out["title"]
    )

    out["volume"] = _to_int(yaml_meta.get("volume"))
    out["number"] = _to_int(yaml_meta.get("number"))
    out["year"] = _to_int(yaml_meta.get("year"))

    author = yaml_meta.get("author")
    if isinstance(author, (list, tuple)):
        out["author"] = ", ".join(str(a).strip() for a in author if a)
    elif isinstance(author, str):
        out["author"] = author.strip()
    else:
        out["author"] = ""

    perm = yaml_meta.get("permission")
    out["permission"] = bool(perm) if perm is not None else None

    # Canonical genre structure: extract raw fields and let normalize_genre_meta
    # perform the canonicalization and fallback logic.
    corpus = yaml_meta.get("corpus")
    subject_area = yaml_meta.get("subject_area")
    kw_raw = yaml_meta.get("keywords")
    if isinstance(kw_raw, (list, tuple)):
        keywords = [str(k).strip() for k in kw_raw if k is not None and str(k).strip()]
    elif isinstance(kw_raw, str):
        # keep free-form strings as a single-item list
        keywords = [kw_raw.strip()] if kw_raw.strip() else []
    else:
        keywords = []

    out["corpus"] = str(corpus).strip() if corpus else None
    out["subject_area"] = str(subject_area).strip() if subject_area else None
    out["keywords"] = keywords if keywords else None

    # If YAML provided an explicit 'genre' field, forward it unchanged so the
    # central normalizer can resolve genre1/genre2/genre3 consistently.
    if "genre" in yaml_meta:
        out["genre"] = yaml_meta["genre"]

    return normalize_genre_meta(out)


def _read_yaml_front_matter_from_path(
    p: Path, max_bytes: int = 64 * 1024
) -> dict[str, Any] | None:
    """
    Safely read only the first `max_bytes` bytes of file p and parse YAML front matter.
    Returns mapping or None.
    """
    try:
        with open(p, "r", encoding="utf-8") as fh:
            head = fh.read(max_bytes)
    except Exception:
        return None
    return _parse_yaml_front_matter(head)


def _folder_has_yaml_majority(folder: Path, ext: str, sample_n: int = 20) -> bool:
    """
    Sample up to sample_n files with extension `ext` in `folder` and return True if a
    majority contain YAML front matter. Uses random.sample for the sample.
    """
    files = sorted(folder.glob(f"*{ext}"))
    if not files:
        return False
    n = min(sample_n, len(files))
    try:
        sampled = random.sample(files, n)
    except Exception:
        sampled = files[:n]
    found = 0
    total = 0
    for p in sampled:
        try:
            if _read_yaml_front_matter_from_path(p):
                found += 1
        except Exception:
            pass
        total += 1
    return total > 0 and (found * 2 >= total)


def _blocks_have_header_attrs(blocks: list[Any]) -> bool:
    """
    Return True if any Header block in `blocks` contains a non-empty Attr
    (id, classes or keyvals). Conservative and fast: reads only header attrs.
    """
    if not blocks:
        return False
    for blk in blocks:
        if not isinstance(blk, Header):
            continue
        try:
            attr = blk[1]
        except Exception:
            continue
        if not attr:
            continue
        try:
            _id = attr[0] if len(attr) > 0 else None
            classes = attr[1] if len(attr) > 1 else None
            keyvals = attr[2] if len(attr) > 2 else None
        except Exception:
            _id = None
            classes = None
            keyvals = None

        if _id:
            return True
        if classes:
            if isinstance(classes, (list, tuple)):
                if any(bool(c) for c in classes):
                    return True
            elif isinstance(classes, str) and classes.strip():
                return True
        if keyvals:
            if isinstance(keyvals, list) and keyvals:
                return True
    return False


def _canonical_pre_path_for(src_path: Path, pre_suffix: str = ".pre") -> Path:
    """
    Return a deterministic preprocessed companion path inside CACHE_DIR for `src_path`.

    Example: .cache/pandoc_ast/<hash>_<stem>.pre.md
    Uses the resolved absolute posix path of the source to avoid collisions across folders.
    """
    key = xxhash.xxh64(
        str(Path(src_path).resolve().as_posix()).encode("utf-8")
    ).hexdigest()
    pre_name = f"{key}_{Path(src_path).stem}{pre_suffix}{Path(src_path).suffix}"
    return CACHE_DIR / pre_name


def get_pandoc_cache_path_for(src_path: Path, text: str | None = None) -> Path:
    """
    Compute the .pkl pandoc-AST cache path that parse_markdown_blocks_cached
    would use for the canonical preprocessed companion of `src_path`.

    - If the canonical pre file exists, uses its stat (mtime_ns,size) to build the key.
    - Else, if `text` is provided, uses the sanitized text hash as a fallback key.
    - Else falls back to a deterministic hash of the resolved src_path.
    """
    pre_path = _canonical_pre_path_for(src_path)
    try:
        if pre_path.exists():
            st = pre_path.stat()
            key = f"{pre_path.resolve().as_posix()}|{st.st_mtime_ns}|{st.st_size}"
        else:
            if text is not None:
                sanitized_text = _remove_html_br_tags(remove_cjk_whitespace(text or ""))
                key = xxhash.xxh64(sanitized_text.encode("utf-8")).hexdigest()
            else:
                # Deterministic fallback if we cannot stat or have no text
                key = xxhash.xxh64(
                    str(Path(src_path).resolve().as_posix()).encode("utf-8")
                ).hexdigest()
    except Exception:
        if text is not None:
            sanitized_text = _remove_html_br_tags(remove_cjk_whitespace(text or ""))
            key = xxhash.xxh64(sanitized_text.encode("utf-8")).hexdigest()
        else:
            key = xxhash.xxh64(
                str(Path(src_path).resolve().as_posix()).encode("utf-8")
            ).hexdigest()

    return CACHE_DIR / (xxhash.xxh64(key.encode("utf-8")).hexdigest() + ".pkl")


def is_pandoc_cache_fresh(src_path: Path) -> bool:
    """
    Return True if a pandoc AST .pkl exists for the canonical pre file of src_path.
    This checks the cache key computed from the current canonical pre file's stat.
    """
    try:
        cache_p = get_pandoc_cache_path_for(src_path)
        return cache_p.exists()
    except Exception:
        return False


def clear_pandoc_cache(
    older_than_seconds: int | None = None, pattern: str = "*.pkl"
) -> int:
    """
    Remove cache artifacts from CACHE_DIR.

    - If older_than_seconds is provided, only remove files older than that threshold.
    - pattern controls which file types to remove (default *.pkl). To remove pre files use pattern='*_*.pre.*'
    Returns the number of files removed.
    """
    removed = 0
    threshold = (
        None if older_than_seconds is None else (time.time() - older_than_seconds)
    )
    for p in CACHE_DIR.glob(pattern):
        try:
            if threshold is not None:
                if p.stat().st_mtime > threshold:
                    continue
            p.unlink()
            removed += 1
        except Exception:
            logging.debug("Failed to delete cache file %s", p, exc_info=True)
    return removed


# Precompiled ordered mapping rules (was inline in _normalize_section_name)
_ORDERED_SECTION_RULES_RAW = [
    (r"(?:理論仮説|仮説モデル)", "hypotheses"),
    (r"(?:推計モデル|モデル設定)", "methods"),
    (r"(?:頑健性の確認|ロバストネス確認)", "discussion"),
    (r"(?:むすび)", "conclusion"),
    (
        r"^(?:I{1,3}|IV|V|VI{0,3}|X{1,2})\s*(?:\.\s*)?(?:はじめに|緒言|序論)",
        "introduction",
    ),
    (r"(?:位置づけ|位置付け)", "position"),
    (r"(?:結果[お及びと]?考察|考察と結果|まとめと考察)", "results_discussion"),
    (r"(?:キーワード|keywords?)", "keywords"),
    (r"(?:要旨|概要|抄録|アブストラクト|abstract|要約)", "abstract"),
    (r"(?:目次|contents?)", "toc"),
    (r"(?:先行研究|関連研究|既往研究|研究史)", "literature_review"),
    (
        r"(?:はじめ|初め|始め|序論|緒言|導入|序|問題の所在|問題設定|問題提起|背景|研究の背景|研究背景|問題意識|研究目的|研究の目的|本稿の目的|目的と背景|目的)",
        "introduction",
    ),
    (
        r"(?:方法論|方法|手法|研究方法|研究の方法|調査方法|調査の方法|分析方法|実験方法|推定方法|対象と方法|データと方法|モデルと推定方法|研究デザイン|手続き|実験手続|実験の手続き|実験の概要|調査概要|調査の概要|調査(?:地(?:および)?)?(?:および調査)?の概要)",
        "methods",
    ),
    (
        r"(?:使用するデータ|データ説明|データと変数|分析データ|データ|資料|史料|材料|文献調査)",
        "data",
    ),
    (r"(?:結果の解釈|解釈)", "discussion"),
    (r"(?:推定結果|実験結果|分析結果|結果の概要|結果|検証結果)", "results"),
    (
        r"(?:ディスカッション|総合考察|調査結果の考察|要約と議論|分析と考察|議論|考察|分析)",
        "discussion",
    ),
    (
        # include both kana (かえて) and kanji (代えて) variants for むすびにかえて
        r"(?:結論と展望|結語|結言|結びにかえて|結びに代えて|むすびに(?:か|代)えて|むすび|終わりに|おわりに|最後に|小括|小結|結び|まとめ|結論)",
        "conclusion",
    ),
    (
        r"(?:今後の研究課題|本稿の限界と今後の課題|本研究の限界と今後の課題|結論と今後の課題|まとめと課題|おわりに今後の課題|今後の課題|展望)",
        "future_work",
    ),
    (r"(?:謝辞|感謝)", "acknowledgments"),
    (r"(?:注及び引用・参考文献|補注|注記|註|注)", "notes"),
    (
        r"(?:引用・?参考文献|参照文献|文献一覧|文献表|文献リスト|参考引用文献|引用資料|引用文献|参考文献|英語文献|巻末資料|\A文献\Z)",
        "references",
    ),
    (r"(?:付表|補遺|付記|付録)", "appendix"),
    (r"(?:執筆者紹介|プロフィール|著者紹介)", "author_bio"),
    (r"(?:仮説の検証|仮説設定|仮説提示|仮説)", "hypotheses"),
]
ORDERED_SECTION_RULES = [
    (re.compile(p, flags=re.IGNORECASE), cat) for p, cat in _ORDERED_SECTION_RULES_RAW
]
ORDERED_CAT_PRIORITY = {cat: idx for idx, (rx, cat) in enumerate(ORDERED_SECTION_RULES)}

# Child->parent subsumption map (transitive parents)
_CATEGORY_PARENTS: dict[str, set[str]] = {
    "hypotheses": {"introduction"},
    "position": {"introduction"},
}

PROPAGATE_SUBTYPES = frozenset(
    {
        # structural/section-level subtypes we want to surface on parents
        "hypotheses",
        "future_work",
        "acknowledgments",
        "notes",
        # "author_bio",
        # "keywords",
        "abstract",
        # "toc",
        # "references",
        "appendix",
        "literature_review",
    }
)


def _expand_with_parents(cats: set[str] | None) -> set[str]:
    """Return cats unioned with any parent categories (transitively)."""
    if not cats:
        return set()
    out = set(cats)
    changed = True
    while changed:
        changed = False
        for c in list(out):
            parents = _CATEGORY_PARENTS.get(c)
            if parents:
                new = parents - out
                if new:
                    out.update(new)
                    changed = True
    return out


# How many characters of a heading we consider "the head" for cheap regex matching
HEAD_MATCH_WINDOW = 30

# Regex for inline enumerated hypothesis-like headings, e.g. "仮説 1: ...", "Hypothesis 2: ..."
_ENUM_HYPOTHESIS_RE = re.compile(
    r"^\s*(?:仮説|Hypothesis|hypothesis)\s*[：:\s]*([0-9０-９①-⑳ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫ一二三四五六七八九十百千]+)"
)

# patterns used in merge_synthetic_nodes: precompile once
_PARA_NUM_PREFIX_PATTERNS = [
    re.compile(r"^\s*(\d+(?:[.\-．][\d０-９]+)*\.?)[\s　]*(.+)$"),
    re.compile(r"^\s*[（(]\s*(\d+(?:[.\-．]\d+)*)\s*[)）][\s　]*(.+)$"),
    re.compile(r"^\s*([①-⑳])[\s　]*(.+)$"),
    # ASCII Roman numerals: "I", "II", "III", optionally "I." etc.
    re.compile(r"^\s*([IVXLCDMivxlcdm]+)(?:[.\．])?[\s　]*(.+)$"),
    # Fullwidth/Unicode Roman numerals: Ⅰ, Ⅱ, Ⅲ, ...
    re.compile(r"^\s*([ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫ]+)[\s　]*(.+)$"),
    re.compile(r"^\s*([一二三四五六七八九十百千]+)[\.\．]?[\s　]*(.+)$"),
]


def _cat_as_set(cat: set[str] | str | None) -> set[str]:
    if not cat:
        return set()
    return cat if isinstance(cat, set) else {cat}


def _pick_primary_category(cats: set[str] | None) -> str | None:
    """
    Pick one canonical category from a set deterministically.
    Prefer a parent category (if present) and otherwise use ORDERED_CAT_PRIORITY.
    """
    if not cats:
        return None
    cats_set = set(cats)
    # Prefer parent-class candidates if present
    parent_values = {p for parents in _CATEGORY_PARENTS.values() for p in parents}
    parent_candidates = parent_values & cats_set
    if parent_candidates:
        return min(parent_candidates, key=lambda c: ORDERED_CAT_PRIORITY.get(c, 10**9))
    # fallback: use ordered-priority
    return min(cats_set, key=lambda c: ORDERED_CAT_PRIORITY.get(c, 10**9))


def _propagate_child_categories(node: SectionNode) -> set[str] | None:
    """
    Propagate selected matched_subtypes from immediate children into parents.

    Rules:
    - Preserve each node.matched_subtypes (the original rule matches).
    - Compute node.subtypes := matched_subtypes(node) U (union of matched_subtypes(child) intersect PROPAGATE_SUBTYPES)
      i.e. only surface a child's structural subtypes (like 'hypotheses', 'future_work', etc.) into the parent.
    - Compute node.category by expanding node.subtypes with parents via _expand_with_parents.
    This avoids promoting content-level labels such as 'results' up into an 'experiment' parent while still surfacing
    structural labels like 'future_work' and 'hypotheses' on parent nodes.
    """
    # Snapshot original matched subtypes for all nodes (safe if matched_subtypes is None)
    orig_matched: dict[int, set[str]] = {}

    def _collect(n: SectionNode) -> None:
        val = getattr(n, "matched_subtypes", None)
        orig_matched[id(n)] = set(val) if val else set()
        for c in n.children:
            _collect(c)

    _collect(node)

    def _apply(n: SectionNode) -> None:
        # start with node's own matched_subtypes (original rule matches)
        subs = set(orig_matched.get(id(n), set()))
        # union immediate children's matched_subtypes but only the propagateable ones
        for c in n.children:
            child_matched = orig_matched.get(id(c), set())
            if child_matched:
                subs.update(child_matched & PROPAGATE_SUBTYPES)
        n.subtypes = subs if subs else None
        n.category = _expand_with_parents(n.subtypes) if n.subtypes else None
        for c in n.children:
            _apply(c)

    # Apply propagation starting from the root's children so the document/root node
    # does not absorb aggregated child categories. This prevents collapse_same_category
    # from clearing top-level section categories by making them equal to the root.
    for child in node.children:
        _apply(child)
    return node.category


# number/item detection used elsewhere
NUMERIC_COMPONENTS_RE = re.compile(
    r"[\d０-９]+|[①-⑳]|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫ]+|[一二三四五六七八九十百千]+"
)
_ARABIC_DIGITS_RE = re.compile(r"[\d０-９]+")
_CIRCLED_RE = re.compile(r"[①-⑳]")
_ROMAN_RE = re.compile(r"[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫ]+")
_KANJI_NUM_RE = re.compile(r"[一二三四五六七八九十百千]+")
# Skip headings that are clearly table/figure captions (e.g. "表1", "表 2", "図1.", "図 ①")
_TABLE_OR_FIG_HEADING_RE = re.compile(
    r"^\s*(?:表|図)(?:\s*(?:[0-9０-９一二三四五六七八九十百千]+|[①-⑳]|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫ])|[\s\.\-．\u3000:：])"
)

# Canonical default set of section-level labels that we exclude from exported sections.
# This is used as the default value for `section_exclude` across the module.
SECTION_EXCLUDE_DEFAULT = frozenset(
    {
        "unknown",
        "keywords",
        "abstract",
        "toc",
        "acknowledgments",
        "notes",
        "references",
        "appendix",
        "author_bio",
        "author",
        "title",
        "subtitle",
    }
)

# Div attribute classes that indicate block-level metadata to drop entirely.
# Start with the canonical section-exclude set and add a few Div-specific tokens.
_DIV_ATTR_EXCLUDE = set(SECTION_EXCLUDE_DEFAULT) | {
    "author-bio",  # alternative dash-form
    "caption",
    "figcaption",
}

# Precompute a normalized form (lowercased, '-' -> '_') for fast membership tests.
_DIV_ATTR_EXCLUDE_NORMALIZED = {c.replace("-", "_").lower() for c in _DIV_ATTR_EXCLUDE}


def _looks_like_heading_candidate(s: str) -> bool:
    """
    Heuristic check whether a line fragment is likely a section heading.

    Returns True for short/noun-phrase-like fragments or strings that
    contain heading punctuation. Returns False for long, sentence-like
    lines or list items (circled numbers, many commas, sentence-final
    punctuation, typical verb endings).
    """
    if not s:
        return False
    s = s.strip()
    # Reject clearly sentence-like or list-like fragments
    if len(s) > 120:
        return False
    if "。" in s or "！" in s or "？" in s:
        return False
    # If the fragment contains circled list markers, treat as list content
    if _CIRCLED_RE.search(s):
        return False
    # Too many commas (ASCII or Japanese) -> likely a sentence with clauses
    if s.count("、") + s.count(",") > 1:
        return False
    # Heuristic: ends with common verb/adjective endings typical of sentences
    if re.search(
        r"(する|します|した|ている|である|ある|いる|ます|られる|です|だった|ています)$",
        s,
    ):
        return False
    # Short fragments are likely headings
    if len(s) <= 40:
        return True
    # If the fragment contains explicit heading separators or colons, accept
    if ":" in s or "：" in s or "―" in s or "—" in s:
        return True
    # Conservative default: not a heading
    return False


def remove_cjk_whitespace(text: str) -> str:
    """Remove whitespace only when between two CJK characters (including fullwidth spaces)."""
    return _CJK_SPACE_RE.sub(r"\1\2", text)


def _remove_html_br_tags(text: str) -> str:
    """
    Remove HTML <br> tags (including <br/>, <br />) and <sup>...</sup> tags
    (keeping the inner content) from markdown text before handing it to pandoc.

    Examples:
    >>> _remove_html_br_tags("a<br>b")
    'ab'
    >>> _remove_html_br_tags("x<BR/>y")
    'xy'
    >>> _remove_html_br_tags("foo<sup>1</sup>bar")
    'foo1bar'
    """
    if not text:
        return text
    # remove <br> variants first, convert to newline
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    # remove opening/closing <sup> tags but preserve inner content; accept attributes
    text = re.sub(r"</?sup\b[^>]*>", "", text, flags=re.IGNORECASE)
    return text


def _preprocess_and_write_markdown(
    file_path: Path, text: str, pre_suffix: str = ".pre"
) -> tuple[Path, str]:
    """
    Preprocess markdown text into a canonical form for pandoc and persist it to CACHE_DIR.

    The preprocessed companion is stored as:
      CACHE_DIR / "<xxhash(path.resolve())>_<stem>.pre<suffix>"

    If the file already exists and content matches, the existing path is returned.
    Writes are atomic via a tmp file and os.replace.
    """
    pre_text = _remove_html_br_tags(remove_cjk_whitespace(text or ""))
    pre_path = _canonical_pre_path_for(file_path, pre_suffix=pre_suffix)

    try:
        if pre_path.exists():
            try:
                if pre_path.read_text(encoding="utf-8") == pre_text:
                    return pre_path, pre_text
            except Exception:
                # fall through to rewrite if read fails
                pass

        # Atomic, thread-unique temp file to avoid races
        tmp = pre_path.with_suffix(pre_path.suffix + f".tmp{threading.get_ident()}")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(pre_text)
        os.replace(tmp, pre_path)
        logging.debug("Wrote preprocessed markdown to cache: %s", pre_path)
    except Exception:
        logging.debug(
            "Failed to write preprocessed markdown for %s", file_path, exc_info=True
        )

    return pre_path, pre_text


def _is_mostly_cjk(text: str, threshold: float = 0.4) -> bool:
    """
    Check if text has at least threshold proportion of CJK characters.

    Returns True if the text should be kept (>= threshold CJK), False if it should be filtered out.
    """
    if not text.strip():
        return False

    # Count CJK characters (Chinese, Japanese, Korean)
    cjk_count = 0
    total_chars = 0

    for char in text:
        # Skip whitespace and common punctuation in the count
        if char.isspace() or char in '.,;:!?()[]{}""\'「」『』（）【】〈〉《》':
            continue
        total_chars += 1

        # Check if character is in CJK Unicode ranges
        code_point = ord(char)
        if (
            (0x4E00 <= code_point <= 0x9FFF)  # CJK Unified Ideographs
            or (0x3400 <= code_point <= 0x4DBF)  # CJK Extension A
            or (0x3040 <= code_point <= 0x309F)  # Hiragana
            or (0x30A0 <= code_point <= 0x30FF)  # Katakana
            or (0xAC00 <= code_point <= 0xD7AF)  # Hangul Syllables
            or (0x1100 <= code_point <= 0x11FF)  # Hangul Jamo
            or (0x3130 <= code_point <= 0x318F)  # Hangul Compatibility Jamo
            or (0xFF00 <= code_point <= 0xFFEF)  # Halfwidth and Fullwidth Forms
        ):
            cjk_count += 1

    if total_chars == 0:
        return False

    # Allow short blocks with at least one CJK character
    if total_chars <= 10 and cjk_count > 0:
        return True

    cjk_ratio = cjk_count / total_chars

    # Additional heuristic: if text is long and has some CJK, it's probably academic text with citations
    if len(text) > 100 and cjk_count > 20:  # At least 20 CJK characters in long text
        return True

    return cjk_ratio >= threshold


# global counter for plaintext sentences/docs
_id_counter = count()


def _read_genres(tsv: Path) -> pl.DataFrame:
    return pl.read_csv(
        tsv,
        separator="\t",
        has_header=False,
        new_columns=["code", "genre"],
        schema={"code": pl.String, "genre": pl.String},
    )


def _read_sources(tsv: Path) -> pl.DataFrame:
    # Peek at first line to decide format
    with open(tsv, "r", encoding="utf-8") as f:
        header_line = f.readline().strip()

    headers = header_line.split("\t")
    # Heuristic: treat as headered if all fields are alpha or contain non-ASCII (Japanese)
    has_header = all(h.isalpha() or any(ord(c) > 127 for c in h) for h in headers)

    if "ファイル名" in header_line or "basename" in headers or has_header:
        # Headered multi-col format (Japanese or English headers)
        # Force all columns to String to avoid Polars dtype inference errors
        col_names = pl.read_csv(tsv, separator="\t", has_header=True, n_rows=0).columns
        df = pl.read_csv(
            tsv,
            separator="\t",
            quote_char=None,
            has_header=True,
            schema_overrides={col: pl.String for col in col_names},
        )
        # Map Japanese column names to expected English keys
        COLUMN_MAP = {
            "ファイル名": "basename",
            "basename": "basename",
            "タイトル": "journal_title",
            "journal_title": "journal_title",
            "著者": "author",
            "author": "author",
            "論文誌": "journal_name",
            "journal_name": "journal_name",
            "vol": "vol",
            "no": "no",
            "ページ数": "pages",
            "url": "url",
            "year": "year",
            "permission": "permission",
            "ジャンル": "genre",
        }
        df = df.rename({col: COLUMN_MAP.get(col, col) for col in df.columns})

        logging.debug(f"Post-rename columns: {df.columns}")
        # Normalise columns: make sure we always have basename, journal_title, author, journal_name, year, permission
        # Add default year/permission if missing
        if "year" not in df.columns:
            df = df.with_columns(pl.lit("2010").alias("year"))
        if "permission" not in df.columns:
            df = df.with_columns(pl.lit("1").alias("permission"))
        return df

    else:
        # Original 5-col format with no header
        return pl.read_csv(
            tsv,
            separator="\t",
            quote_char=None,
            has_header=False,
            new_columns=["journal_title", "author", "year", "basename", "permission"],
            schema={
                "journal_title": pl.String,
                "author": pl.String,
                "year": pl.String,
                "basename": pl.String,
                "permission": pl.String,
            },
        )


def list_folder_sources(
    folder: Path,
    ext: Literal[".txt", ".md"] | None = None,
    strict_metadata: bool = True,
) -> pl.DataFrame:
    """
    Quickly list files and metadata for a folder without parsing file contents.

    - If "sources.tsv" (and optional "genres.tsv") exists in the folder, the
      returned table mirrors that metadata augmented with a best-effort `path`
      (full path to .md/.txt if present) and an `exists` flag.
    - If no sources.tsv is present, the function performs a fast scan of
      "*.md" and "*.txt" files and returns a row per file with basic inferred
      metadata (title set to the filename).

    This function is intentionally lightweight and does not call pandoc or
    read whole file contents, making it suitable for interactive file selectors
    in notebooks. You can pass the resulting dataframe's basename list as the
    `file_filter` argument to parse_plain_folder_to_tuples to avoid re-scanning
    metadata files.

    :param folder: directory containing .md/.txt files and optional sources.tsv
    :param ext: optional hint (".md" or ".txt") used when choosing candidate paths
    :param strict_metadata: preserved for API parity (not used for quick listing)
    :return: Polars DataFrame with columns including
             ['basename','path','preprocessed_path','preprocessed_exists','title','journal_title','author','year','genre','permission','exists','ext']

    Example:
    >>> from pathlib import Path
    >>> df = list_folder_sources(Path("."))  # doctest: +SKIP
    >>> isinstance(df, pl.DataFrame)
    True
    """
    folder = Path(folder)
    md_files = sorted(folder.glob("*.md"))
    txt_files = sorted(folder.glob("*.txt"))

    # Choose a preferred extension when attempting to resolve missing files
    if ext is None:
        if md_files and not txt_files:
            ext_for = ".md"
        elif txt_files and not md_files:
            ext_for = ".txt"
        elif md_files and txt_files:
            ext_for = ".md"
        else:
            ext_for = None
    else:
        ext_for = ext

    # Build a fast mapping stem -> Path preferring .md over .txt when both exist
    path_map: dict[str, Path] = {}
    for p in md_files:
        path_map[p.stem] = p
    for p in txt_files:
        if p.stem not in path_map:
            path_map[p.stem] = p

    genres_tsv = folder / "genres.tsv"
    sources_tsv = folder / "sources.tsv"

    rows: list[dict[str, Any]] = []

    if genres_tsv.exists() and sources_tsv.exists():
        try:
            folder_genre = (
                _read_genres(genres_tsv)["genre"][0]
                if len(_read_genres(genres_tsv)) > 0
                else None
            )
        except Exception:
            folder_genre = None

        try:
            sources_df = _read_sources(sources_tsv)
        except Exception:
            sources_df = pl.DataFrame()

        for r in sources_df.to_dicts():
            basename = r.get("basename") or r.get("ファイル名") or ""
            if not basename:
                continue

            path_obj = path_map.get(basename)
            if path_obj is None:
                # Try preferred extension if available
                if ext_for:
                    candidate = folder / f"{basename}{ext_for}"
                    if candidate.exists():
                        path_obj = candidate
                # Fallback to checking both .md and .txt
                if path_obj is None:
                    m = folder / f"{basename}.md"
                    t = folder / f"{basename}.txt"
                    if m.exists():
                        path_obj = m
                    elif t.exists():
                        path_obj = t

            exists = bool(path_obj)
            # Candidate preprocessed companion (do not write/parse here; just expose the expected path)
            if path_obj is not None:
                pre_path_obj = _canonical_pre_path_for(path_obj)
            else:
                pre_path_obj = None
                if ext_for:
                    pre_path_obj = folder / f"{basename}.pre{ext_for}"

            # Try to pick up per-file YAML front matter to enrich/override metadata for quick listing.
            yaml_info: dict[str, Any] | None = None
            if path_obj and Path(path_obj).exists():
                yaml_meta = _read_yaml_front_matter_from_path(Path(path_obj))
                if not yaml_meta and pre_path_obj and Path(pre_path_obj).exists():
                    yaml_meta = _read_yaml_front_matter_from_path(Path(pre_path_obj))
                if yaml_meta:
                    try:
                        yaml_info = _yaml_meta_to_info(yaml_meta, Path(path_obj))
                    except Exception:
                        yaml_info = None

            # Use YAML-provided values when present, otherwise fall back to TSV values / folder defaults
            if yaml_info:
                title_val = yaml_info.get("title") or ""
                author_val = yaml_info.get("author") or ""
                year_val = yaml_info.get("year")
                # human-readable genre string for quick listing: prefer full path, else genre1
                genre_val = (
                    yaml_info.get("genre_path") or yaml_info.get("genre") or None
                )
            else:
                title_val = (
                    r.get("journal_title")
                    or r.get("title")
                    or r.get("journal_name")
                    or ""
                )
                author_val = r.get("author") or ""
                year_raw = r.get("year")
                try:
                    year_val = (
                        int(year_raw)
                        if year_raw is not None and str(year_raw).isdigit()
                        else None
                    )
                except Exception:
                    year_val = None
                genre_val = r.get("genre") or folder_genre

            permission = r.get("permission") or ""

            rows.append(
                {
                    "basename": str(basename),
                    "path": str(path_obj) if path_obj is not None else "",
                    "preprocessed_path": str(pre_path_obj)
                    if pre_path_obj is not None
                    else "",
                    "preprocessed_exists": bool(pre_path_obj and pre_path_obj.exists()),
                    # Titles
                    "title": str(title_val or ""),
                    "journal_title": str(
                        (yaml_info or {}).get("journal_title")
                        or r.get("journal_title")
                        or ""
                    ),
                    "journal_name": str(
                        (yaml_info or {}).get("journal_name")
                        or r.get("journal_name")
                        or ""
                    ),
                    # Bib/meta
                    "author": str(author_val or ""),
                    "year": year_val,
                    "volume": (yaml_info or {}).get("volume"),
                    "number": (yaml_info or {}).get("number"),
                    # Genre fields (preserve structure)
                    "corpus": (yaml_info or {}).get("corpus"),
                    "subject_area": (yaml_info or {}).get("subject_area"),
                    "keywords": (yaml_info or {}).get("keywords"),
                    "genre": (yaml_info or {}).get("genre")
                    if yaml_info
                    else (str(genre_val) if genre_val is not None else None),
                    "genre1": (yaml_info or {}).get("genre1"),
                    "genre2": (yaml_info or {}).get("genre2"),
                    "genre3": (yaml_info or {}).get("genre3"),
                    "genre_path": (yaml_info or {}).get("genre_path"),
                    # Permission and housekeeping
                    "permission": (yaml_info or {}).get("permission")
                    if yaml_info
                    else (
                        permission
                        if isinstance(permission, bool)
                        else (permission == "1")
                    ),
                    "exists": exists,
                    "ext": path_obj.suffix if path_obj is not None else None,
                }
            )
    else:
        # No sources.tsv: return one row per discovered file (fast filesystem scan)
        for stem, p in sorted(path_map.items()):
            pre_path = _canonical_pre_path_for(p)

            # Try to pick up per-file YAML front matter to enrich/override metadata for quick listing.
            yaml_info: dict[str, Any] | None = None
            if Path(p).exists():
                yaml_meta = _read_yaml_front_matter_from_path(Path(p))
                if not yaml_meta and pre_path.exists():
                    yaml_meta = _read_yaml_front_matter_from_path(pre_path)
                if yaml_meta:
                    try:
                        yaml_info = _yaml_meta_to_info(yaml_meta, Path(p))
                    except Exception:
                        yaml_info = None

            if yaml_info:
                title_val = yaml_info.get("title") or ""
                author_val = yaml_info.get("author") or ""
                year_val = yaml_info.get("year")
                # human-readable genre string for quick listing: prefer full path, else genre1
                genre_val = (
                    yaml_info.get("genre_path") or yaml_info.get("genre") or None
                )
            else:
                title_val = str(stem)
                author_val = ""
                year_val = None
                genre_val = None

            rows.append(
                {
                    "basename": str(stem),
                    "path": str(p),
                    "preprocessed_path": str(pre_path),
                    "preprocessed_exists": bool(pre_path.exists()),
                    # Titles
                    "title": str(title_val or ""),
                    "journal_title": str((yaml_info or {}).get("journal_title") or ""),
                    "journal_name": str((yaml_info or {}).get("journal_name") or ""),
                    # Bib/meta
                    "author": str(author_val or ""),
                    "year": year_val,
                    "volume": (yaml_info or {}).get("volume"),
                    "number": (yaml_info or {}).get("number"),
                    # Genre fields (preserve structure)
                    "corpus": (yaml_info or {}).get("corpus"),
                    "subject_area": (yaml_info or {}).get("subject_area"),
                    "keywords": (yaml_info or {}).get("keywords"),
                    "genre": (yaml_info or {}).get("genre")
                    if yaml_info
                    else (str(genre_val) if genre_val is not None else None),
                    "genre1": (yaml_info or {}).get("genre1"),
                    "genre2": (yaml_info or {}).get("genre2"),
                    "genre3": (yaml_info or {}).get("genre3"),
                    "genre_path": (yaml_info or {}).get("genre_path"),
                    # Permission and housekeeping
                    "permission": (yaml_info or {}).get("permission"),
                    "exists": True,
                    "ext": p.suffix,
                }
            )

    return pl.DataFrame(rows)


def _extract_text_from_blocks(blocks, in_note: bool = False) -> str:
    """Manually extract text from pandoc AST blocks as fallback."""

    text_parts = []

    def extract_from_inlines(inlines):
        """Extract text from inline elements."""
        parts = []
        for inline in inlines:
            # Preserve intra-paragraph line breaks
            name = getattr(inline, "__class__", type(inline)).__name__
            if name in ("SoftBreak", "LineBreak"):
                parts.append("\n")
                continue

            if name == "Note":
                # Footnotes are ignored to prevent their content from being merged with the main text.
                continue

            if name == "RawInline":
                # RawInline is a tuple of (Format, Text). We want the text.
                try:
                    parts.append(inline[1])
                except Exception:
                    # Be forgiving if shape is unexpected
                    pass
                continue

            if isinstance(inline, Str):
                parts.append(inline[0])  # Str(Text) - access the text content
            elif isinstance(inline, Space):
                parts.append(" ")
            # Handle other inline types that might contain nested inlines
            elif hasattr(inline, "__iter__") and len(inline) > 0:
                try:
                    # For Link, the text is in the second element: Link(Attr, [Inline], Target)
                    if (
                        hasattr(inline, "__class__")
                        and inline.__class__.__name__ == "Link"
                    ):
                        nested_inlines = inline[1]  # Second element is [Inline]
                    else:
                        nested_inlines = inline[
                            -1
                        ]  # Usually the last element is [Inline]
                    if isinstance(nested_inlines, list):
                        parts.append(extract_from_inlines(nested_inlines))
                except (IndexError, TypeError):
                    pass
        return "".join(parts)

    def extract_from_block(block):
        """Extract text from a single block."""
        block_name = block.__class__.__name__

        if block_name == "Para":
            return extract_from_inlines(block[0])
        elif block_name == "Header":
            return extract_from_inlines(block[2])
        elif block_name == "Plain":
            return extract_from_inlines(block[0])
        elif block_name in ("BulletList", "OrderedList", "BlockQuote"):
            # For list-like blocks, extract raw text and then sentence-split it.
            raw_text_parts = []

            def _recursive_extract(b):
                """Recursively extract text from nested block structures."""
                if isinstance(b, (Para, Plain)):
                    raw_text_parts.append(extract_from_inlines(b[0]))
                elif isinstance(b, Header):
                    raw_text_parts.append(extract_from_inlines(b[2]))
                elif isinstance(b, (list, tuple)):
                    for item in b:
                        _recursive_extract(item)
                elif hasattr(b, "__class__"):
                    # For list-like blocks, recurse on their content (usually at index 0)
                    block_name = b.__class__.__name__
                    if block_name in ("BulletList", "OrderedList", "BlockQuote", "Div"):
                        try:
                            content = b[0] if block_name != "Div" else b[1]
                            _recursive_extract(content)
                        except (IndexError, TypeError):
                            pass

            _recursive_extract(block)

            raw_text = "\n".join(raw_text_parts)
            if not raw_text.strip():
                return ""

            sat = get_sat_model()
            sentences = sat.split(raw_text)
            return "\n".join(sentences)

        return ""

    for block in blocks:
        text = extract_from_block(block)
        if text.strip():
            text_parts.append(text.strip())

    return "\n\n".join(text_parts)


# Lightweight singleton for wtpsplit SaT sentence splitter.
_sat_model: SaT | None = None
_sat_model_lock = threading.Lock()


def get_sat_model() -> SaT:
    """Return a process-global SaT model instance (singleton)."""
    global _sat_model
    if _sat_model is None:
        with _sat_model_lock:
            if _sat_model is None:
                logging.info("Loading SaT model for sentence splitting...")
                _sat_model = SaT("sat-3l")
                try:
                    import torch

                    if torch.cuda.is_available():
                        _sat_model.half().to("cuda")
                        logging.info("SaT model moved to CUDA.")
                except ImportError:
                    logging.info("PyTorch not found, SaT model will run on CPU.")
    assert _sat_model is not None
    return _sat_model


def _get_md_parse_executor() -> tuple[
    type[ThreadPoolExecutor | ProcessPoolExecutor], int
]:
    """
    Get the executor class and max workers for parallel pandoc parsing.

    Defaults to ThreadPoolExecutor with os.cpu_count() workers, which is safer
    in multi-threaded environments (like notebooks) and efficient for I/O-bound
    tasks like calling the pandoc executable.

    Configuration can be overridden with environment variables:
    - DM_PARSE_USE_PROCESSES=1: Use ProcessPoolExecutor instead of ThreadPoolExecutor.
    - DM_PARSE_WORKERS=N: Set the number of workers.
    """
    use_processes = os.getenv("DM_PARSE_USE_PROCESSES", "0").lower() in (
        "1",
        "true",
        "yes",
    )
    Executor = ProcessPoolExecutor if use_processes else ThreadPoolExecutor

    # A more conservative default for workers
    default_workers = os.cpu_count() or 4
    max_workers = int(os.getenv("DM_PARSE_WORKERS", str(default_workers)))

    return Executor, max_workers


def _normalize_section_name(
    section: str,
) -> tuple[set[str] | None, str | None, set[str] | None]:
    """
    Normalize section names for consistent grouping, using lemma-based matching
    and NER/POS heuristics instead of only brittle hard-coded surface forms.

    Returns a tuple: (set of categories, number_prefix, original matched subtypes)
    """
    # normalize intermittent CJK spacing so matching uses the same space-normalized form
    text = remove_cjk_whitespace(section.strip())
    if not text:
        return None, None, None

    # allow single-character headings if they match an ordered rule (e.g. "注" -> notes)
    if len(text) == 1:
        head_slice = text[:HEAD_MATCH_WINDOW]
        for rx, _ in ORDERED_SECTION_RULES:
            if rx.match(text) or rx.search(head_slice):
                break
        else:
            return None, None, None

    # Extract leading numbering (Arabic, circled, kanji, parenthesis, roman, multi-level, etc.)
    m = NUMBER_PREFIX_RE.match(text)
    if m:
        number_prefix = m.group(0).strip()
        text_core = text[m.end() :].strip()

        # If the body begins with another numeric token (e.g. "3. 1 ..."), treat that
        # token as a continuation of the numeric prefix so "3. 1" becomes "3.1".
        # This allows headings written with a space after separators to be interpreted
        # as multi-component prefixes instead of a single top-level prefix + body.
        lead_match = re.match(
            r"^\s*([（(]?\s*[0-9０-９①-⑳ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫ一二三四五六七八九十百千]+(?:[.\-．－\s]+[0-9０-９①-⑳ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫ一二三四五六七八九十百千]+)*)",
            text_core,
        )
        if lead_match:
            lead = lead_match.group(1)
            # normalize whitespace/fullwidth space inside the leading numeric token
            lead_norm = re.sub(r"[\s　]+", "", lead)
            # Combine with existing prefix. If prefix already ends with a separator, append directly.
            if number_prefix.endswith((".", "-", "．", "－")):
                number_prefix = number_prefix + lead_norm
            else:
                # Insert a dot between components when the original prefix ends with a digit-like char
                if re.search(
                    r"[\d０-９①-⑳ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫ一二三四五六七八九十百千]$", number_prefix
                ):
                    number_prefix = number_prefix + "." + lead_norm
                else:
                    number_prefix = number_prefix + lead_norm
            # remove the consumed part from text_core
            text_core = text_core[lead_match.end() :].strip()
    else:
        number_prefix = None
        text_core = text

    # Fast regex mapping: try to map known section names before invoking GiNZA.
    # Use search() so patterns can match anywhere in the heading (not only at the start).
    # Also try the original heading text as a fallback (in case number-prefix slicing removed context).
    # Restrict rule matching to the start of the heading; prefer anchored matches
    # and choose the most specific (longest) matched substring when multiple rules match.
    head_portion = text_core[:HEAD_MATCH_WINDOW]
    candidates: list[tuple[str, int, bool]] = []

    for rx, cat in ORDERED_SECTION_RULES:
        # Try anchored match on the slice that excludes numbering first
        m = rx.match(text_core)
        anchored = True
        if m is None:
            # Also try anchored on the original text (fallback)
            m = rx.match(text)
            anchored = True if m else False

        if m is None:
            # Finally allow searching only within the head portion. Be permissive when the
            # match is preceded by a hiragana particle or whitespace/punctuation (e.g.
            # "家族の位置づけ"), but continue to reject matches embedded in Kanji/katakana
            # compounds (e.g. avoid matching "導入" inside "再導入") or ASCII/alnum runs.
            m = rx.search(head_portion)
            if m:
                start_pos = m.start()
                if start_pos > 0:
                    prev_char = head_portion[start_pos - 1]
                    # Block if prev char is Kanji (CJK ideograph), Katakana, or ASCII/fullwidth alnum/digit.
                    if re.match(
                        r"[A-Za-z0-9０-９\u4E00-\u9FFF\u3400-\u4DBF\u30A0-\u30FF\uFF10-\uFF19]",
                        prev_char,
                    ):
                        m = None
                    else:
                        anchored = False
                else:
                    anchored = False

        if m:
            span_len = m.end() - m.start()
            candidates.append((cat, span_len, anchored))
    if candidates:
        # prefer the longest matched substring (more specific)
        max_len = max(c[1] for c in candidates)
        best = [c for c in candidates if c[1] == max_len]
        # if any best candidate is anchored, restrict to anchored ones
        if any(c[2] for c in best):
            best = [c for c in best if c[2]]
        matched_cats = {c[0] for c in best}

        # If "references" was matched along with other categories, prefer it as the
        # explicit section label rather than leaving it as a subtype of the other label.
        # This prevents headings that mention references together with other terms from
        # being mis-classified as e.g. future_work with references as a subtype.
        if "references" in matched_cats and len(matched_cats) > 1:
            matched_cats = {"references"}

        return _expand_with_parents(matched_cats), number_prefix, matched_cats

    # No regex match. Fallback to simple heuristics without spaCy.
    # If there is a number prefix, we assume it's a valid (but unknown) section.
    if number_prefix:
        return None, number_prefix, None

    # If no number prefix and no regex match, it's likely not a section heading.
    logging.debug(f"Unmapped section heading '{section}' → dropping section metadata")
    return None, None, None


def parse_plain_folder_to_tuples(
    folder: Path,
    ext: Literal[".txt", ".md"] | None = None,  # Optional: auto‐detect
    strict_metadata: bool = True,
    file_filter: set[str] | None = None,
    section_exclude: set[str] | None = None,  # normalized section names to discard
    skip_section_classification: bool | None = None,
) -> Iterator[tuple[str, dict[str, Any]]]:
    """
    Yield (text, metadata) tuples for each segment in folder files.
    Compatible with the JSONL sentence iterator format.
    """
    # Canonicalize incoming folder so behavior does not depend on current working directory
    folder = Path(folder).expanduser().resolve(strict=False)
    logging.info(
        f"Parsing folder: {folder} (exists={folder.exists()}, cwd={Path.cwd()})"
    )
    logging.info(f"strict_metadata parameter: {strict_metadata}")

    # Default to skipping these sections if not specified (shared canonical set)
    if section_exclude is None:
        section_exclude = set(SECTION_EXCLUDE_DEFAULT)

    # Auto-detect file extension if not specified
    if ext is None:
        # Fast top-level scan (case-insensitive suffix matching)
        try:
            entries = list(folder.iterdir())
        except Exception:
            entries = []
        md_files = [p for p in entries if p.is_file() and p.suffix.lower() == ".md"]
        txt_files = [p for p in entries if p.is_file() and p.suffix.lower() == ".txt"]

        # Fallback: if none found at top level, try a recursive search (rglob)
        if not md_files and not txt_files and folder.exists():
            md_files = list(folder.rglob("*.md"))
            txt_files = list(folder.rglob("*.txt"))

        if md_files and not txt_files:
            ext = ".md"
            logging.info("Auto-detected .md files")
        elif txt_files and not md_files:
            ext = ".txt"
            logging.info("Auto-detected .txt files")
        elif md_files and txt_files:
            # Prefer .md when both exist
            ext = ".md"
            logging.info("Both .md and .txt files found, preferring .md")
        else:
            # No candidates found — keep a short directory preview for debugging
            ext = ".txt"
            logging.info(
                "No files found, defaulting to .txt extension. Directory preview (up to 20 entries): %s",
                [p.name for p in entries[:20]],
            )

    logging.info(f"Using file extension: {ext}")

    genres_tsv = folder / "genres.tsv"
    sources_tsv = folder / "sources.tsv"

    logging.info("Checking for metadata files:")
    logging.info(f"  genres.tsv exists: {genres_tsv.exists()} at {genres_tsv}")
    logging.info(f"  sources.tsv exists: {sources_tsv.exists()} at {sources_tsv}")

    # Decide whether to prefer per-file YAML front matter over sources.tsv (if present).
    prefer_yaml = False
    try:
        prefer_yaml = _folder_has_yaml_majority(folder, ext)
    except Exception:
        prefer_yaml = False
    if prefer_yaml:
        logging.info(
            "Detected majority YAML front matter in folder; per-file YAML metadata will be used (over sources.tsv)."
        )

    if genres_tsv.exists() and sources_tsv.exists():
        logging.info("TAKING METADATA PATH - both files exist")
        logging.debug("genres.tsv content preview:")
        with open(genres_tsv, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < 3:
                    logging.debug(f"  Line {i}: {line.strip()}")
        logging.debug("sources.tsv content preview:")
        with open(sources_tsv, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < 3:
                    logging.debug(f"  Line {i}: {line.strip()}")
        genres_df = _read_genres(genres_tsv)
        sources_df = _read_sources(sources_tsv)
        logging.info(f"Loaded {len(genres_df)} genres and {len(sources_df)} sources")

        folder_genre = genres_df["genre"][0] if len(genres_df) > 0 else None
        if not folder_genre:
            raise ValueError(f"No genre found in {genres_tsv}")
        logging.info(f"Using genre '{folder_genre}' for all sources in this folder")

        sources_df = sources_df.with_columns(
            pl.lit(folder_genre).alias("genre")
        ).filter(pl.col("basename").is_not_null() & (pl.col("basename") != ""))

        logging.info(f"Processing {len(sources_df)} sources with valid basenames")
        logging.debug(f"Sample source row: {sources_df.head(1).to_dicts()}")

        md_tasks = []
        for info in sources_df.to_dicts():
            basename = info.get("basename")
            if not basename or basename == "null":
                logging.warning(f"Skipping row with invalid basename: {basename}")
                continue

            # Apply file filter if provided
            if file_filter is not None and basename not in file_filter:
                continue

            txt = folder / f"{basename}{ext}"
            if not txt.exists():
                logging.warning(f"Text file not found: {txt}")
                continue

            if ext == ".md":
                if prefer_yaml:
                    # Pass minimal base info; _process_md_file will parse YAML and validate when strict_metadata=True.
                    task_info: dict[str, Any] = {"basename": basename}
                else:
                    # enforce strict metadata from sources.tsv as before
                    if not info.get("journal_title"):
                        raise ValueError(
                            f"Missing title for basename {info.get('basename')}"
                        )
                    # Year is optional: log debug when missing/invalid but do not abort.
                    if not info.get("year") or not str(info.get("year")).isdigit():
                        logging.debug(
                            "Row %s has missing/invalid year: %s (continuing without year)",
                            info.get("basename"),
                            info.get("year"),
                        )
                    if not info.get("genre"):
                        raise ValueError(
                            f"Missing genre for basename {info.get('basename')}"
                        )
                    task_info = info

                # defer markdown parsing to the thread pool; pass strict_metadata flag so the worker can validate YAML results
                md_tasks.append(
                    (
                        txt,
                        task_info,
                        section_exclude,
                        strict_metadata,
                        skip_section_classification,
                    )
                )
            else:
                # .txt processing: always use the sources.tsv row (no YAML normally expected in .txt)
                if not info.get("journal_title"):
                    raise ValueError(
                        f"Missing title for basename {info.get('basename')}"
                    )
                # Year is optional: log debug when missing/invalid but do not abort.
                if not info.get("year") or not str(info.get("year")).isdigit():
                    logging.debug(
                        "Row %s has missing/invalid year: %s (continuing without year)",
                        info.get("basename"),
                        info.get("year"),
                    )
                if not info.get("genre"):
                    raise ValueError(
                        f"Missing genre for basename {info.get('basename')}"
                    )
                yield (txt.read_text(encoding="utf-8"), info)

        # Run Markdown parsing in parallel (configurable worker count). Immediately
        # release per-file result references and invoke gc to avoid transient memory accumulation.
        if md_tasks:
            Executor, max_workers = _get_md_parse_executor()
            logging.info(
                "Processing %d markdown tasks (using %s with %d workers)",
                len(md_tasks),
                Executor.__name__,
                max_workers,
            )
            with Executor(max_workers=max_workers) as executor:
                for results in executor.map(_process_md_file, md_tasks):
                    for plain_text, meta_dict in results:
                        yield (plain_text, meta_dict)
                    # free memory held by the results list and ask GC to reclaim
                    try:
                        del results
                    except Exception:
                        pass
                    gc.collect()
    else:
        logging.info("No sources.tsv/genres.tsv detected; processing files directly")
        # Fast filesystem scan for markdown / text files
        md_files = sorted(folder.glob("*.md"))
        txt_files = sorted(folder.glob("*.txt"))

        # Decide which set to process based on earlier autodetected ext
        chosen_md: list[Path] = []
        chosen_txt: list[Path] = []
        if ext == ".md":
            chosen_md = md_files
        elif ext == ".txt":
            chosen_txt = txt_files
        else:
            # prefer markdown when present
            if md_files:
                chosen_md = md_files
            else:
                chosen_txt = txt_files

        # Yield plain .txt files directly (no pandoc/pipeline required)
        if chosen_txt:
            for p in chosen_txt:
                if file_filter is not None and p.stem not in file_filter:
                    continue
                try:
                    yaml_meta = _read_yaml_front_matter_from_path(p)
                    info = (
                        _yaml_meta_to_info(yaml_meta, p)
                        if yaml_meta
                        else {"basename": p.stem, "title": p.stem}
                    )
                except Exception:
                    info = {"basename": p.stem, "title": p.stem}
                try:
                    text = p.read_text(encoding="utf-8")
                except Exception:
                    logging.warning("Failed to read %s, skipping", p)
                    continue
                yield (text, info)

        # For markdown files, schedule parsing workers that call _process_md_file
        md_tasks: list[tuple[Path, dict[str, Any], set[str], bool]] = []
        if chosen_md:
            for p in chosen_md:
                if file_filter is not None and p.stem not in file_filter:
                    continue
                try:
                    yaml_meta = _read_yaml_front_matter_from_path(p)
                    if yaml_meta:
                        task_info = _yaml_meta_to_info(yaml_meta, p)
                    else:
                        if strict_metadata:
                            logging.warning(
                                "Skipping %s: missing YAML metadata and no sources.tsv present",
                                p,
                            )
                            continue
                        task_info = {"basename": p.stem, "title": p.stem}
                except Exception:
                    if strict_metadata:
                        logging.warning(
                            "Skipping %s: failed to read YAML metadata",
                            p,
                            exc_info=True,
                        )
                        continue
                    task_info = {"basename": p.stem, "title": p.stem}

                md_tasks.append(
                    (
                        p,
                        task_info,
                        section_exclude,
                        strict_metadata,
                        skip_section_classification,
                    )
                )

            if md_tasks:
                Executor, max_workers = _get_md_parse_executor()
                logging.info(
                    "Processing %d markdown tasks (using %s with %d workers)",
                    len(md_tasks),
                    Executor.__name__,
                    max_workers,
                )
                with Executor(max_workers=max_workers) as executor:
                    for results in executor.map(_process_md_file, md_tasks):
                        for plain_text, meta_dict in results:
                            yield (plain_text, meta_dict)
                        try:
                            del results
                        except Exception:
                            pass
                        gc.collect()


def yield_section_texts(
    root: SectionNode,
    info: dict[str, Any],
    title: str,
    section_exclude: set[str],
    text: str,
):
    """
    Recursively yield (plain_text, meta_dict) for each section node.
    """
    year = info.get("year")
    genre1 = (
        str(info.get("genre1") or info.get("corpus") or info.get("genre") or "").strip()
        or None
    )
    genre2 = str(info.get("genre2") or info.get("subject_area") or "").strip() or None
    kw = info.get("keywords")
    if isinstance(kw, str):
        kw = _split_keywords(kw)
    elif isinstance(kw, list):
        kw = [str(k).strip() for k in kw if str(k).strip()]
    else:
        kw = None
    genre3 = kw[0] if kw else None
    genre_path = " / ".join([s for s in [genre1, genre2] if s])

    visited = set()

    def _earliest_excluded_descendant_start(node: SectionNode) -> int | None:
        """Return the smallest start_char of any descendant whose category intersects section_exclude."""
        if not section_exclude:
            return None
        earliest: int | None = None
        stack = list(node.children)
        while stack:
            ch = stack.pop()
            if ch.category:
                ch_cats = ch.category if isinstance(ch.category, set) else {ch.category}
                if ch_cats & section_exclude:
                    if earliest is None or ch.start_char < earliest:
                        earliest = ch.start_char
            if ch.children:
                stack.extend(ch.children)
        return earliest

    def _recurse(
        node, parent_cat=None, parent_num=None, parent_primary=None, path=None
    ):
        if id(node) in visited:
            return
        visited.add(id(node))

        effective_category = node.category if node.category is not None else parent_cat
        effective_number = node.num_prefix or parent_num

        # Determine the node's primary category and build the hierarchical path
        primary_cat = (
            _pick_primary_category(effective_category) if effective_category else None
        )
        section_label = primary_cat or "unknown"

        new_path = list(path or [])
        if primary_cat:
            new_path = new_path + [primary_cat]

        # canonical document bounds
        doc_len = len(text)
        node_start = node.start_char if node.start_char is not None else 0
        sec_end = node.end_char if node.end_char is not None else doc_len

        # clamp to earliest excluded descendant start (unchanged semantics)
        excluded_start = _earliest_excluded_descendant_start(node)
        if excluded_start is not None and excluded_start > node_start:
            sec_end = min(sec_end, excluded_start)

        # compute body_start: remove header text robustly when present
        body_start = node_start
        header_text = None
        if getattr(node, "block", None) is not None and isinstance(node.block, Header):
            header_text = remove_cjk_whitespace(
                _extract_text_from_blocks([node.block]).strip()
            )
            if header_text:
                sec_slice = text[node_start:sec_end]
                if sec_slice.startswith(header_text):
                    body_start = node_start + len(header_text)
                else:
                    # try to find header_text somewhere in the section slice
                    idx = sec_slice.find(header_text)
                    if idx != -1:
                        body_start = node_start + idx + len(header_text)
                    else:
                        # try a broader search in the document range for robustness
                        pos = text.find(header_text, node_start, sec_end)
                        if pos != -1:
                            body_start = pos + len(header_text)
                        else:
                            # last-resort: drop the first short line only when it looks like a numeric prefix
                            try:
                                first_line = (
                                    sec_slice.splitlines()[0]
                                    if sec_slice.splitlines()
                                    else ""
                                )
                                if (
                                    first_line
                                    and len(first_line) < 200
                                    and NUMBER_PREFIX_RE.match(first_line.strip())
                                ):
                                    body_start = node_start + len(first_line)
                            except Exception:
                                pass

        # Build parent body by subtracting labeled child spans; keep unlabeled descendants inside
        labeled_children = [
            c for c in node.children if c.start_char is not None and c.category
        ]
        labeled_children.sort(key=lambda c: c.start_char or 0)

        pieces: list[str] = []
        cursor = body_start
        for ch in labeled_children:
            s = max(cursor, ch.start_char or cursor)
            e = min(ch.end_char if ch.end_char is not None else sec_end, sec_end)
            if s > cursor:
                pieces.append(text[cursor:s])
            cursor = max(cursor, e)
        if cursor < sec_end:
            pieces.append(text[cursor:sec_end])

        own_text = remove_cjk_whitespace(
            "\n\n".join(p.strip() for p in pieces if p.strip())
        )

        # If we did not obtain text via slicing (rare), fall back to block-extraction for non-header blocks
        if (
            not own_text
            and getattr(node, "block", None) is not None
            and not isinstance(node.block, Header)
        ):
            try:
                fb = _extract_text_from_blocks([node.block]).strip()
                fb = remove_cjk_whitespace(fb)
                own_text = fb
            except Exception:
                own_text = ""

        # Yield this node's exclusive body only (no duplication of children)
        if own_text and _is_mostly_cjk(own_text, threshold=0.6):
            if not (section_exclude and section_label in section_exclude):
                # only yield if this node is labeled, or it has no labeled ancestor
                is_labeled = bool(node.category)
                has_labeled_parent = bool(parent_primary)
                if is_labeled or not has_labeled_parent:
                    sentence_id = next(_id_counter)
                    meta_dict = {
                        "title": title,
                        "subtitle": info.get("subtitle"),
                        "genre": genre1,
                        "ジャンル": genre1,
                        "genre1": genre1,
                        "genre2": genre2,
                        "genre3": genre3,
                        "keywords": kw if kw else None,
                        "genre_path": genre_path if genre_path else None,
                        "corpus": (info.get("corpus") or genre1),
                        "subject_area": genre2,
                        "basename": str(info.get("basename", "")),
                        "author": str(info.get("author", ""))
                        if info.get("author") is not None
                        else None,
                        "year": year,
                        "sentence_id": sentence_id,
                        "paragraph_id": None,
                        "section": section_label,
                        "section_number": effective_number,
                        "section_level": getattr(node, "level", None),
                        "section_parent": parent_primary,
                        "section_path": new_path if new_path else None,
                        # section-level subtype information (returned as sorted lists or None)
                        "section_subtypes": sorted(list(node.subtypes))
                        if getattr(node, "subtypes", None)
                        else None,
                        "section_matched_subtypes": sorted(list(node.matched_subtypes))
                        if getattr(node, "matched_subtypes", None)
                        else None,
                    }
                    logging.debug(
                        f"Yielding section: section={section_label}, num={effective_number}, "
                        f"text[:10]='{own_text[:10]}...'"
                    )
                    yield (own_text, meta_dict)

        # Recurse into children (they will yield their exclusive text)
        for child in node.children:
            yield from _recurse(
                child,
                effective_category,
                effective_number,
                primary_cat,
                new_path,
            )

    for top in root.children:
        yield from _recurse(top, None, None, None, None)


def parse_plain_folder(
    folder: Path,
    nlp: Language,
    ext: Literal[".txt", ".md"] | None = None,  # Make optional here too
    strict_metadata: bool = True,
) -> Iterator[Doc]:
    """
    Yield spaCy Docs for each segment in folder files.
    Uses the same processing pipeline as JSONL loading with caching.
    """

    # Create cache key based on folder contents
    hash_obj = xxhash.xxh64()
    genres_tsv = folder / "genres.tsv"
    sources_tsv = folder / "sources.tsv"

    if genres_tsv.exists():
        with open(genres_tsv, "rb") as f:
            hash_obj.update(f.read())
    if sources_tsv.exists():
        with open(sources_tsv, "rb") as f:
            hash_obj.update(f.read())

    # If ext is None, auto-detect in parse_plain_folder_to_tuples, but for cache key, try both
    ext_for_hash = ext
    if ext_for_hash is None:
        md_files = list(folder.glob("*.md"))
        txt_files = list(folder.glob("*.txt"))
        if md_files and not txt_files:
            ext_for_hash = ".md"
        elif txt_files and not md_files:
            ext_for_hash = ".txt"
        elif md_files and txt_files:
            ext_for_hash = ".md"
        else:
            ext_for_hash = ".txt"
    text_files = sorted(folder.glob(f"*{ext_for_hash}"))
    for txt_file in text_files:
        if txt_file.exists():
            hash_obj.update(txt_file.name.encode("utf-8"))
            with open(txt_file, "rb") as f:
                hash_obj.update(f.read())

    hash_obj.update((ext_for_hash or "").encode("utf-8"))
    hash_obj.update(str(strict_metadata).encode("utf-8"))
    cache_hash = hash_obj.hexdigest()[:16]

    # Create a temporary JSONL file for caching
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    temp_jsonl = cache_dir / f"plain_folder_{folder.name}_{cache_hash}.jsonl"

    # Only create JSONL if it doesn't exist
    if not temp_jsonl.exists():
        logging.info(f"Creating temporary JSONL for caching: {temp_jsonl}")

        with open(temp_jsonl, "wb") as f:
            current_title = None
            current_genre = None
            sentences = []

            for text, metadata in parse_plain_folder_to_tuples(
                folder, ext, strict_metadata
            ):
                title = metadata["title"]
                genre = metadata["genre"]

                if title != current_title and sentences:
                    # Flush previous group
                    record = {
                        "title": current_title,
                        "genre": current_genre,
                        "sentences": sentences,
                    }
                    f.write(orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE))
                    sentences.clear()

                current_title = title
                current_genre = genre
                sentences.append(text)

            if sentences:
                record = {
                    "title": current_title,
                    "genre": current_genre,
                    "sentences": sentences,
                }
                f.write(orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE))

    parser = CorpusParser(temp_jsonl, nlp)
    yield from parser.stream()


def parse_md_to_section_nodes(
    file_path: Path,
    title: str | None = None,
    section_exclude: set[str] | None = None,
    skip_classification: bool | None = None,
) -> list[SectionNode]:
    """
    Parse a single Markdown (.md) file into SectionNode objects with body_text attached.

    :param file_path: Path to the Markdown file.
    :param title: Optional document title; if None, defaults to empty string.
    :param section_exclude: Optional set of normalized section categories to exclude.
    :return: List of SectionNode objects (flat list in document order), each with .body_text
    """
    if section_exclude is None:
        section_exclude = set(SECTION_EXCLUDE_DEFAULT)

    # Support being passed either the original markdown path (e.g. "doc.md")
    # or the preprocessed companion (e.g. "doc.pre.md"). If a preprocessed
    # path is provided but missing, try to synthesize it from the original.
    pre_suffix = ".pre"
    file_path = Path(file_path)
    if file_path.exists():
        text = file_path.read_text(encoding="utf-8")
        pre_path, pre_text = _preprocess_and_write_markdown(file_path, text)
    else:
        # If caller passed a preprocessed candidate that does not yet exist,
        # attempt to build it from the original (remove ".pre" from the stem).
        if file_path.name.endswith(f"{pre_suffix}{file_path.suffix}"):
            orig_stem = file_path.stem[: -len(pre_suffix)]
            orig_path = file_path.parent / f"{orig_stem}{file_path.suffix}"
            if orig_path.exists():
                orig_text = orig_path.read_text(encoding="utf-8")
                pre_path, pre_text = _preprocess_and_write_markdown(
                    orig_path, orig_text
                )
            else:
                raise FileNotFoundError(
                    f"Neither preprocessed path {file_path!s} nor original {orig_path!s} exist"
                )
        else:
            raise FileNotFoundError(f"File not found: {file_path!s}")

    if title is None:
        title = ""

    # Parse Markdown into Pandoc AST blocks using the preprocessed file/text.
    blocks, sanitized_text = parse_markdown_blocks_cached(pre_path, pre_text)

    # Detect first H1 + H2 as title + subtitle (same heuristic as in _process_md_file)
    subtitle = None
    if (
        len(blocks) >= 2
        and isinstance(blocks[0], Header)
        and blocks[0][0] == 1
        and isinstance(blocks[1], Header)
        and blocks[1][0] == 2
    ):

        def _extract_inlines_as_text(inlines):
            parts = []
            for inline in inlines:
                if isinstance(inline, Str):
                    parts.append(inline[0])
                elif isinstance(inline, Space):
                    parts.append(" ")
                elif hasattr(inline, "__iter__") and len(inline) > 0:
                    try:
                        nested = inline[-1]
                        if isinstance(nested, list):
                            parts.append(_extract_inlines_as_text(nested))
                    except (IndexError, TypeError):
                        pass
            return "".join(parts)

        first_h1 = remove_cjk_whitespace(_extract_inlines_as_text(blocks[0][2]).strip())
        first_h2 = remove_cjk_whitespace(_extract_inlines_as_text(blocks[1][2]).strip())
        if not title:
            title = first_h1
        subtitle = first_h2

    # Extract headings as SectionNodes (pass subtitle)
    real_nodes = extract_section_nodes(
        blocks, title, subtitle, sanitized_text, skip_classification=skip_classification
    )

    # Determine annotated mode by flag or Header Attrs
    annotated_mode = bool(
        (skip_classification is True) or _blocks_have_header_attrs(blocks)
    )

    if annotated_mode:
        all_nodes = list(real_nodes)
    else:
        # Merge synthetic nodes using sanitized text (so offsets align with the AST)
        all_nodes = merge_synthetic_nodes(real_nodes, sanitized_text)
        # Assign hierarchical numbering for inline enumerations such as "仮説 1: ..."
        _assign_enumerated_hypotheses(all_nodes)
        _normalize_numeric_sequences(all_nodes)
        _promote_numeric_section_prefixes(all_nodes, base_level=2)
        _smooth_level_jumps(all_nodes)

    # Promote/lift canonical top-level sections (references, appendix, toc, abstract, notes, etc.)
    if not annotated_mode:
        TOP_LEVEL_SECTIONS = {
            "references",
            "appendix",
            "toc",
            "abstract",
            "acknowledgments",
            "author_bio",
            "notes",
        }
        try:
            for n in all_nodes:
                ms = getattr(n, "matched_subtypes", None)
                if ms and set(ms) & TOP_LEVEL_SECTIONS:
                    if (n.level or 0) != 2:
                        logging.debug(
                            "Promoting canonical top-level heading to level 2: start=%s raw=%r",
                            getattr(n, "start_char", None),
                            getattr(n, "raw_text", None),
                        )
                        n.level = 2
        except Exception:
            logging.debug("Failed to promote canonical headings", exc_info=True)

    # Build section tree for hierarchy and end_char assignment
    root = build_section_tree(all_nodes, len(sanitized_text))

    # Lift any remaining canonical sections that were nested into the root (preserve doc order).
    if not annotated_mode:
        TOP_LEVEL_SECTIONS = {
            "references",
            "appendix",
            "toc",
            "abstract",
            "acknowledgments",
            "author_bio",
            "notes",
        }
        try:
            moved: list[SectionNode] = []

            def _collect_and_lift(parent: SectionNode) -> None:
                for ch in list(parent.children):
                    ms = getattr(ch, "matched_subtypes", None)
                    if ms and set(ms) & TOP_LEVEL_SECTIONS:
                        parent.children.remove(ch)
                        ch.level = 2
                        moved.append(ch)
                    else:
                        _collect_and_lift(ch)

            _collect_and_lift(root)
            if moved:
                root.children.extend(moved)
                root.children.sort(key=lambda n: n.start_char or 0)
                assign_end_chars(root, len(sanitized_text))
        except Exception:
            logging.debug(
                "Failed to lift canonical top-level sections post-tree-build",
                exc_info=True,
            )

    # Propagate child categories up to parents and collapse identical categories
    _propagate_child_categories(root)
    collapse_same_category(root)

    # Attach body_text to each node (exclusive of children and excluded descendants).
    def _earliest_excluded_descendant_start(node: SectionNode) -> int | None:
        if not section_exclude:
            return None
        earliest: int | None = None
        stack = list(node.children)
        while stack:
            ch = stack.pop()
            if ch.category:
                ch_cats = ch.category if isinstance(ch.category, set) else {ch.category}
                if ch_cats & section_exclude:
                    if earliest is None or (
                        ch.start_char is not None and ch.start_char < earliest
                    ):
                        earliest = ch.start_char
            if ch.children:
                stack.extend(ch.children)
        return earliest

    for node in all_nodes:
        if node.end_char is None:
            continue
        node_start = node.start_char if node.start_char is not None else 0

        # clamp end against excluded descendants so excluded sections are not included
        exclusive_doc_end = node.end_char
        excluded_start = _earliest_excluded_descendant_start(node)
        if excluded_start is not None and excluded_start > node_start:
            exclusive_doc_end = min(exclusive_doc_end, excluded_start)

        # Determine body_start (skip header text when present)
        body_start = node_start
        if getattr(node, "block", None) is not None and isinstance(node.block, Header):
            header_text = remove_cjk_whitespace(
                _extract_text_from_blocks([node.block]).strip()
            )
            if header_text:
                sec_slice = sanitized_text[node_start:exclusive_doc_end]
                if sec_slice.startswith(header_text):
                    body_start = node_start + len(header_text)
                else:
                    idx = sec_slice.find(header_text)
                    if idx != -1:
                        body_start = node_start + idx + len(header_text)
                    else:
                        pos = sanitized_text.find(
                            header_text, node_start, exclusive_doc_end
                        )
                        if pos != -1:
                            body_start = pos + len(header_text)
                        else:
                            try:
                                first_line = (
                                    sec_slice.splitlines()[0]
                                    if sec_slice.splitlines()
                                    else ""
                                )
                                if (
                                    first_line
                                    and len(first_line) < 200
                                    and NUMBER_PREFIX_RE.match(first_line.strip())
                                ):
                                    body_start = node_start + len(first_line)
                            except Exception:
                                pass

        # Subtract labeled child spans; keep unlabeled descendants within parent body_text
        labeled_children = [
            c for c in node.children if c.start_char is not None and c.category
        ]
        labeled_children.sort(key=lambda c: c.start_char or 0)

        pieces: list[str] = []
        cursor = body_start
        for ch in labeled_children:
            s = max(cursor, ch.start_char or cursor)
            e = min(
                ch.end_char if ch.end_char is not None else exclusive_doc_end,
                exclusive_doc_end,
            )
            if s > cursor:
                pieces.append(sanitized_text[cursor:s])
            cursor = max(cursor, e)
        if cursor < exclusive_doc_end:
            pieces.append(sanitized_text[cursor:exclusive_doc_end])

        node.body_text = remove_cjk_whitespace(
            "\n\n".join(p.strip() for p in pieces if p.strip())
        )

    # Mark nodes that have a labeled ancestor
    has_labeled_ancestor: dict[int, bool] = {}

    def _mark(n: SectionNode, parent_labeled: bool) -> None:
        for ch in n.children:
            curr_has = parent_labeled or bool(n.category)
            has_labeled_ancestor[id(ch)] = curr_has
            _mark(ch, curr_has)

    _mark(root, False)

    # Filter out excluded categories if specified.
    # Keep "unknown" (no-category) nodes when their body_text looks like real CJK content.
    if section_exclude:

        def _node_is_excluded(node: SectionNode) -> bool:
            # If node has an explicit category label, exclude if it intersects the exclude set.
            if node.category:
                cats = (
                    node.category if isinstance(node.category, set) else {node.category}
                )
                return bool(cats & section_exclude)

            # No category: unlabeled descendant of a labeled ancestor should not be its own section
            if has_labeled_ancestor.get(id(node), False):
                return True

            # If caller does not explicitly exclude "unknown", keep the node.
            if "unknown" not in section_exclude:
                return False

            # Otherwise exclude unknown nodes only when they appear non-CJK/empty
            body = getattr(node, "body_text", "") or ""
            if body and _is_mostly_cjk(body, threshold=0.5):
                return False
            return True

        filtered_nodes = [node for node in all_nodes if not _node_is_excluded(node)]
    else:
        filtered_nodes = list(all_nodes)

    return filtered_nodes


def render_md_sections_html(
    file_path: Path,
    title: str | None = None,
    show_unrecognized: bool = True,
) -> str:
    """
    Render a self-contained, notebook-friendly HTML visualization of the section
    structure extracted from a Markdown file.

    The output shows all parsed SectionNodes as vivid, nested boxes. Each box
    displays: level, num, num_prefix, category, subtypes, matched_subtypes and the
    section body text. By default nodes without a recognized category are shown;
    pass show_unrecognized=False to hide them.

    :param file_path: path to the Markdown file to render
    :param title: optional override for document title (if not provided the file
                  name or detected title is used)
    :param show_unrecognized: include nodes that have no detected category
    :return: HTML string (self-contained CSS + markup) suitable for notebook display

    Example:
    >>> from pathlib import Path
    >>> isinstance(render_md_sections_html(Path(__file__)), str)
    True
    """
    # Parse file and reconstruct the same node/tree construction used in
    # parse_md_to_section_nodes so we can render the full tree (recognized and not).
    # Support being passed either the original markdown path or the preprocessed companion.
    pre_suffix = ".pre"
    file_path = Path(file_path)
    if file_path.exists():
        text = file_path.read_text(encoding="utf-8")
        pre_path, pre_text = _preprocess_and_write_markdown(file_path, text)
    else:
        if file_path.name.endswith(f"{pre_suffix}{file_path.suffix}"):
            orig_stem = file_path.stem[: -len(pre_suffix)]
            orig_path = file_path.parent / f"{orig_stem}{file_path.suffix}"
            if orig_path.exists():
                orig_text = orig_path.read_text(encoding="utf-8")
                pre_path, pre_text = _preprocess_and_write_markdown(
                    orig_path, orig_text
                )
            else:
                raise FileNotFoundError(
                    f"Neither preprocessed path {file_path!s} nor original {orig_path!s} exist"
                )
        else:
            raise FileNotFoundError(f"File not found: {file_path!s}")
    if title is None:
        title = ""

    blocks, sanitized_text = parse_markdown_blocks_cached(pre_path, pre_text)

    subtitle = None
    if (
        len(blocks) >= 2
        and isinstance(blocks[0], Header)
        and blocks[0][0] == 1
        and isinstance(blocks[1], Header)
        and blocks[1][0] == 2
    ):
        first_h1 = remove_cjk_whitespace(_extract_text_from_blocks([blocks[0]]).strip())
        first_h2 = remove_cjk_whitespace(_extract_text_from_blocks([blocks[1]]).strip())
        if not title:
            title = first_h1
        subtitle = first_h2

    real_nodes = extract_section_nodes(blocks, title, subtitle, sanitized_text)

    # Respect annotated markdown when headers carry attributes
    annotated_mode = _blocks_have_header_attrs(blocks)

    if annotated_mode:
        all_nodes = list(real_nodes)
    else:
        all_nodes = merge_synthetic_nodes(real_nodes, sanitized_text)
        _assign_enumerated_hypotheses(all_nodes)
        _normalize_numeric_sequences(all_nodes)
        _promote_numeric_section_prefixes(all_nodes, base_level=2)
        _smooth_level_jumps(all_nodes)

    root = build_section_tree(all_nodes, len(sanitized_text))

    if not annotated_mode:
        # Lift canonical top-level sections so they render as siblings rather than nested.
        TOP_LEVEL_SECTIONS = {
            "references",
            "appendix",
            "toc",
            "abstract",
            "acknowledgments",
            "author_bio",
            "notes",
        }
        try:
            moved: list[SectionNode] = []

            def _collect_and_lift(parent: SectionNode) -> None:
                for ch in list(parent.children):
                    ms = getattr(ch, "matched_subtypes", None)
                    if ms and set(ms) & TOP_LEVEL_SECTIONS:
                        parent.children.remove(ch)
                        ch.level = 2
                        moved.append(ch)
                    else:
                        _collect_and_lift(ch)

            _collect_and_lift(root)
            if moved:
                root.children.extend(moved)
                root.children.sort(key=lambda n: n.start_char or 0)
                assign_end_chars(root, len(sanitized_text))
        except Exception:
            logging.debug(
                "Failed to lift canonical top-level sections for rendering",
                exc_info=True,
            )

    _propagate_child_categories(root)
    collapse_same_category(root)

    # Attach body_text to each node (same logic as parse_md_to_section_nodes)
    for node in all_nodes:
        if node.end_char is None:
            continue
        if node.block is not None and isinstance(node.block, Header):
            header_text = remove_cjk_whitespace(
                _extract_text_from_blocks([node.block]).strip()
            )
            section_slice = sanitized_text[node.start_char : node.end_char].lstrip()
            if section_slice.startswith(header_text):
                section_slice = section_slice[len(header_text) :]
            else:
                idx = section_slice.find(header_text)
                if idx != -1:
                    section_slice = section_slice[idx + len(header_text) :]
                else:
                    pos = sanitized_text.find(
                        header_text, node.start_char, node.end_char
                    )
                    if pos != -1:
                        section_slice = sanitized_text[
                            pos + len(header_text) : node.end_char
                        ]
                    else:
                        try:
                            first_line = (
                                section_slice.splitlines()[0]
                                if section_slice.splitlines()
                                else ""
                            )
                            if (
                                first_line
                                and len(first_line) < 200
                                and NUMBER_PREFIX_RE.match(first_line.strip())
                            ):
                                section_slice = section_slice[len(first_line) :]
                        except Exception:
                            pass
            node.body_text = remove_cjk_whitespace(section_slice.strip())
        else:
            try:
                node.body_text = remove_cjk_whitespace(
                    _extract_text_from_blocks([node.block]).strip()
                )
            except Exception:
                node.body_text = remove_cjk_whitespace(
                    sanitized_text[node.start_char : node.end_char].strip()
                )

    from html import escape

    # Build a deterministic category -> color mapping (use a larger base palette and
    # lighten the color for backgrounds so text remains readable).
    base_colors = [
        "#e41a1c",
        "#377eb8",
        "#4daf4a",
        "#984ea3",
        "#ff7f00",
        "#ffff33",
        "#a65628",
        "#f781bf",
        "#999999",
        "#66c2a5",
        "#8da0cb",
        "#e78ac3",
        "#a6d854",
        "#ffd92f",
        "#e5c494",
        "#b3b3b3",
        "#1b9e77",
        "#d95f02",
        "#7570b3",
        "#66a61e",
    ]

    def _lighten_hex(hex_color: str, amount: float = 0.88) -> str:
        """Return a lighter version of hex_color blended toward white by `amount` (0..1)."""
        c = hex_color.lstrip("#")
        if len(c) != 6:
            return hex_color
        try:
            r = int(c[0:2], 16)
            g = int(c[2:4], 16)
            b = int(c[4:6], 16)
        except Exception:
            return hex_color
        r = int(r + (255 - r) * amount)
        g = int(g + (255 - g) * amount)
        b = int(b + (255 - b) * amount)
        return f"#{r:02x}{g:02x}{b:02x}"

    # Collect all distinct category names encountered (flatten sets) in deterministic order.
    cats_seen: set[str] = set()
    categories: list[str] = []
    for n in all_nodes:
        if not n.category:
            continue
        if isinstance(n.category, (set, list, tuple)):
            for c in sorted(n.category):
                if c not in cats_seen:
                    cats_seen.add(c)
                    categories.append(c)
        else:
            if n.category not in cats_seen:
                cats_seen.add(n.category)
                categories.append(n.category)

    # Map category -> (border_color, background_color)
    cat_color_map: dict[str, dict[str, str]] = {}
    for i, cat in enumerate(sorted(categories)):
        color = base_colors[i % len(base_colors)]
        cat_color_map[cat] = {"border": color, "bg": _lighten_hex(color, 0.9)}

    # Neutral styling for unrecognized sections
    UNREC_BORDER = "#888888"
    UNREC_BG = "#f8f8f8"

    css = """<style>
    .md-sections {font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; color:#222; line-height:1.45;}
    .md-sections .header {margin-bottom:10px;}
    .md-sections .legend {margin-bottom:12px; font-size:0.9em;}
    .md-sections .legend .legend-item {display:inline-block; margin-right:8px; padding:4px 8px; border-radius:4px; color:#111;}
    .section {margin:8px 0; border-radius:6px; padding:10px 12px; box-shadow:0 1px 0 rgba(0,0,0,0.03); overflow:auto;}
    .section .meta {margin-bottom:8px; display:flex; flex-wrap:wrap; gap:6px; align-items:center;}
    .tag {display:inline-block; padding:4px 8px; border-radius:999px; font-size:0.82em; color:#fff;}
    .tag.level {background:#333;}
    .tag.small {padding:2px 6px; font-size:0.75em; opacity:0.95; background:rgba(0,0,0,0.08); color:#000;}
    .section.unrecognized {filter:grayscale(6%); opacity:0.95;}
    .section .raw {font-weight:700; margin-bottom:6px;}
    .section .body {white-space:pre-wrap; font-size:0.98em; padding-top:8px; border-top:1px dashed rgba(0,0,0,0.04); margin-top:6px;}

    /* New: dotted boxes around paragraph boundaries inside section bodies */
    .section .body p {
        border: 1px dotted rgba(0,0,0,0.08);
        padding: 8px 10px;
        margin: 8px 0;
        border-radius: 6px;
        background: rgba(255,255,255,0.6);
        line-height: 1.45;
    }

    /* Slightly subtler styling for unrecognized sections */
    .section.unrecognized .body p {
        border-color: rgba(0,0,0,0.06);
        background: rgba(248,248,248,0.6);
    }
    </style>
    """

    html_parts: list[str] = [css]
    html_parts.append(
        f'<div class="md-sections"><div class="header"><strong>{escape(title or file_path.name)}</strong>'
    )
    if subtitle:
        html_parts.append(f' <small style="color:#555">— {escape(subtitle)}</small>')
    html_parts.append("</div>")

    # Legend showing colors per detected category (and an unrecognized marker if needed)
    legend_items: list[str] = []
    for cat in sorted(categories):
        border = cat_color_map[cat]["border"]
        bg = cat_color_map[cat]["bg"]
        legend_items.append(
            f'<span class="legend-item" style="background:{bg}; border-left:6px solid {border};">{escape(cat)}</span>'
        )

    any_unrecognized = any(n.category is None for n in all_nodes)
    if legend_items or any_unrecognized:
        legend_html = "".join(legend_items)
        if any_unrecognized:
            legend_html += f'<span class="legend-item" style="background:{UNREC_BG}; border-left:6px solid {UNREC_BORDER};">unrecognized</span>'
        html_parts.append(f'<div class="legend">{legend_html}</div>')

    def _fmt_field(label: str, s: Any, item_bg: str | None = None) -> str:
        """
        Render a labeled field (e.g. "category:", "subtypes:") followed by one or
        more small pills for the items. Returns an empty string when s is falsy.
        """
        if not s:
            return ""
        items = sorted(list(s)) if isinstance(s, (set, list, tuple)) else [s]
        # label pill uses a subtle neutral background
        label_html = f'<span class="tag small" style="background:rgba(0,0,0,0.06); color:#000; font-weight:600; margin-right:6px;">{escape(str(label))}:</span>'
        if item_bg:
            items_html = "".join(
                f'<span class="tag small" style="background:{item_bg}; color:#000; margin-right:4px;">{escape(str(x))}</span>'
                for x in items
            )
        else:
            items_html = "".join(
                f'<span class="tag small" style="margin-right:4px;">{escape(str(x))}</span>'
                for x in items
            )
        return label_html + items_html

    def _render_node(node: SectionNode) -> str:
        lvl = node.level or 0

        # Determine a single representative category (if any) for coloring
        primary_cat = _pick_primary_category(node.category) if node.category else None

        if primary_cat and primary_cat in cat_color_map:
            border = cat_color_map[primary_cat]["border"]
            bg = cat_color_map[primary_cat]["bg"]
            classes = "section recognized"
        else:
            border = UNREC_BORDER
            bg = UNREC_BG
            classes = "section unrecognized"

        style = f"border-left:6px solid {border}; background:{bg};"
        meta_parts: list[str] = []
        meta_parts.append(f'<span class="tag level">L{lvl}</span>')
        if node.num is not None:
            meta_parts.append(
                f'<span class="tag small">num:{escape(str(node.num))}</span>'
            )
        if node.num_prefix:
            meta_parts.append(
                f'<span class="tag small">prefix:{escape(str(node.num_prefix))}</span>'
            )

        # Display the primary category prominently (with field name). If multiple categories exist,
        # list the extras with a labeled field. Also label subtypes and matched_subtypes.
        if primary_cat:
            meta_parts.append(
                f'<span class="tag small" style="background:{_lighten_hex(border, 0.85)}; color:#000;">category:{escape(str(primary_cat))}</span>'
            )
            if isinstance(node.category, (set, list, tuple)) and len(node.category) > 1:
                extras = [c for c in sorted(node.category) if c != primary_cat]
                if extras:
                    meta_parts.append(_fmt_field("other_categories", extras))
        elif node.category:
            meta_parts.append(_fmt_field("category", node.category))

        if node.subtypes:
            meta_parts.append(_fmt_field("subtypes", node.subtypes))
        if node.matched_subtypes:
            meta_parts.append(_fmt_field("matched_subtypes", node.matched_subtypes))
        meta_html = "".join(meta_parts)

        heading_html = (
            f'<div class="raw">{escape(node.raw_text or "")}</div>'
            if getattr(node, "raw_text", None)
            else ""
        )

        # Compute this node's exclusive body text (exclude all children's spans) so we do not duplicate text.
        doc_len = len(sanitized_text)
        section_start = node.start_char if node.start_char is not None else 0
        section_end = node.end_char if node.end_char is not None else doc_len

        # If the node is a header, skip the header text itself when computing the body start.
        body_start = section_start
        if getattr(node, "block", None) is not None and isinstance(node.block, Header):
            header_text = remove_cjk_whitespace(
                _extract_text_from_blocks([node.block]).strip()
            )
            sec_slice = sanitized_text[section_start:section_end]
            if sec_slice.startswith(header_text):
                body_start = section_start + len(header_text)
            else:
                idx = sec_slice.find(header_text)
                if idx != -1:
                    body_start = section_start + idx + len(header_text)
                else:
                    # broader search in the document range
                    pos = sanitized_text.find(header_text, section_start, section_end)
                    if pos != -1:
                        body_start = pos + len(header_text)

        # Exclusive end is section_end; subtract spans of labeled children to keep unlabeled within parent
        labeled_children = [
            c for c in node.children if c.start_char is not None and c.category
        ]
        labeled_children.sort(key=lambda c: c.start_char or 0)

        pieces: list[str] = []
        cursor = body_start
        for ch in labeled_children:
            s = max(cursor, ch.start_char or cursor)
            e = min(
                ch.end_char if ch.end_char is not None else section_end, section_end
            )
            if s > cursor:
                pieces.append(sanitized_text[cursor:s])
            cursor = max(cursor, e)
        if cursor < section_end:
            pieces.append(sanitized_text[cursor:section_end])

        own_text = remove_cjk_whitespace(
            "\n\n".join(p.strip() for p in pieces if p.strip())
        )

        body_html = ""
        if own_text:
            paras = [escape(p.strip()) for p in own_text.split("\n\n") if p.strip()]
            if paras:
                body_html = (
                    "<div class='body'>"
                    + "".join(f"<p>{p}</p>" for p in paras)
                    + "</div>"
                )

        # Render children (they will render their own exclusive bodies).
        children_html = "".join(
            _render_node(ch)
            for ch in getattr(node, "children", [])
            if (
                ch.category  # always render labeled children
                or (
                    show_unrecognized and primary_cat is None
                )  # render unlabeled only when parent unlabeled
            )
        )
        return f'<div class="{classes} level-{lvl}" style="{style}">{meta_html}{heading_html}{body_html}{children_html}</div>'

    # Render whole document once without duplicating children text:
    doc_len = len(sanitized_text)
    prev_pos = 0
    for top in root.children:
        top_start = top.start_char if top.start_char is not None else prev_pos

        # If there is text between prev_pos and the next node's start, render it as an unrecognized block.
        if top_start > prev_pos:
            gap_text = sanitized_text[prev_pos:top_start].strip()
            if gap_text and show_unrecognized:
                paras = [
                    escape(p.strip())
                    for p in remove_cjk_whitespace(gap_text).split("\n\n")
                    if p.strip()
                ]
                gap_html = "<div class='section unrecognized' style='border-left:6px solid {b}; background:{bg};'>{body}</div>".format(
                    b=UNREC_BORDER,
                    bg=UNREC_BG,
                    body="<div class='body'>"
                    + "".join(f"<p>{p}</p>" for p in paras)
                    + "</div>",
                )
                html_parts.append(gap_html)

        # Render the node (honoring show_unrecognized)
        if show_unrecognized or top.category:
            html_parts.append(_render_node(top))

        prev_pos = top.end_char if top.end_char is not None else top_start

    # Trailing gap after last node
    if prev_pos < doc_len:
        trail = sanitized_text[prev_pos:doc_len].strip()
        if trail and show_unrecognized:
            paras = [
                escape(p.strip())
                for p in remove_cjk_whitespace(trail).split("\n\n")
                if p.strip()
            ]
            trail_html = "<div class='section unrecognized' style='border-left:6px solid {b}; background:{bg};'>{body}</div>".format(
                b=UNREC_BORDER,
                bg=UNREC_BG,
                body="<div class='body'>"
                + "".join(f"<p>{p}</p>" for p in paras)
                + "</div>",
            )
            html_parts.append(trail_html)

    html_parts.append("</div>")
    return "".join(html_parts)
