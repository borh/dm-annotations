"""
dm_extractor: CLI for discourse‐marker extraction and surface‐form export.

Usage:
  dm-extractor extract [OPTIONS] SRC OUT_CSV
  dm-extractor surfaces [OPTIONS] DM_CSV OUT_DIR
  dm-extractor count DM_CSV

Examples:
  dm-extractor extract corpus.jsonl dms.csv
  dm-extractor extract ~/Projects/stjc/JStage-体育学研究/ dms.csv -v
  dm-extractor extract ~/Projects/stjc/JStage-体育学研究/ dms.csv -vv
  dm-extractor surfaces dms.csv out/ --corpus corpus.jsonl --sample -v
  dm-extractor count dms.csv
"""

import logging
from pathlib import Path

import spacy
import typer

from dm_annotations.io.corpus import parse_docs
from dm_annotations.io.export import export_count, export_dms, export_surface_forms

app = typer.Typer(help=__doc__, no_args_is_help=True)


def setup_logging(verbose: int):
    """Set up logging based on verbosity level."""
    if verbose == 0:
        level = logging.WARNING
    elif verbose == 1:
        level = logging.INFO
    else:  # verbose >= 2
        level = logging.DEBUG

    logging.basicConfig(level=level, format="%(levelname)s:%(name)s:%(message)s")


@app.command(
    "extract",
    help="(1/2) Extract DM matches from a JSONL corpus or directory and write CSV.",
)
def extract(
    src: Path = typer.Argument(
        ..., help="Input JSONL corpus or directory with metadata files"
    ),
    out_csv: Path = typer.Argument(..., help="Output CSV file for DM matches"),
    model: str = typer.Option("ja_ginza", help="spaCy model name"),
    cache_batch_size: int = typer.Option(
        100_000, "--batch-size", "-b", help="Sentences per cache file"
    ),
    pipe_batch_size: int = typer.Option(
        800, "--pipe-batch-size", "-p", help="spaCy .pipe batch size"
    ),
    sf_final_filter: bool = typer.Option(
        True,
        "--sf-final-filter/--no-sf-final-filter",
        help="Enable sentence-final bunsetsu filter for SF matches",
    ),
    strict_connectives: bool = typer.Option(
        True,
        "--strict-connectives/--no-strict-connectives",
        help="Require comma immediately after token‐matcher connectives",
    ),
    strict_metadata: bool = typer.Option(
        True,
        "--strict-metadata/--no-strict-metadata",
        help="Require valid metadata (title, year, genre) for directories",
    ),
    section_exclude: list[str] = typer.Option(
        None,
        "--section-exclude",
        help="List of normalized section names to exclude completely (e.g., references, keywords)",
    ),
    sample: float = typer.Option(
        1.0,
        "--sample",
        "-s",
        help="Proportion of data to sample (0.0-1.0, default 1.0 = all data)",
        min=0.0,
        max=1.0,
    ),
    verbose: int = typer.Option(
        0,
        "--verbose",
        "-v",
        count=True,
        help="Increase verbosity (-v for INFO, -vv for DEBUG)",
    ),
) -> None:
    """Extract DM matches using spaCy and export CSV + counts."""
    setup_logging(verbose)

    if sample < 1.0:
        logging.info(f"Sampling {sample:.1%} of input data")

    spacy.prefer_gpu()
    from dm_annotations import load_dm_nlp

    nlp = load_dm_nlp(model, disable=["ner"])
    nlp.get_pipe("dm_extractor").sf_final_filter = sf_final_filter
    nlp.get_pipe("dm_extractor").strict_connectives = strict_connectives

    # Verify the component was added
    if "dm_extractor" not in nlp.pipe_names:
        raise RuntimeError(f"Failed to add dm_extractor to pipeline: {nlp.pipe_names}")

    logging.info(f"Pipeline components: {nlp.pipe_names}")

    if src.is_dir():
        # Handle directory with metadata files
        from dm_annotations.io.text_loader import parse_plain_folder_to_tuples

        # ensure we can sentence‐split text files
        if "sentencizer" not in nlp.pipe_names and "senter" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer", first=True)

        if sample < 1.0:
            # Sample at the file level first to avoid processing all files
            import random

            from dm_annotations.io.text_loader import _read_sources

            sources_tsv = src / "sources.tsv"
            if sources_tsv.exists():
                sources_df = _read_sources(sources_tsv)
                total_files = len(sources_df)
                sample_size = max(1, int(total_files * sample))
                sampled_basenames = set(
                    random.sample(sources_df["basename"].to_list(), sample_size)
                )
                logging.info(
                    f"Sampling {sample_size} files out of {total_files} ({sample:.1%})"
                )
            else:
                # Fallback: sample files directly from filesystem
                all_files = list(src.glob("*.md")) or list(src.glob("*.txt"))
                sample_size = max(1, int(len(all_files) * sample))
                sampled_basenames = set(
                    random.sample([f.stem for f in all_files], sample_size)
                )
                logging.info(
                    f"Sampling {sample_size} files out of {len(all_files)} ({sample:.1%})"
                )

            sentence_tuples = parse_plain_folder_to_tuples(
                src,
                strict_metadata=strict_metadata,
                file_filter=sampled_basenames,
                section_exclude=set(section_exclude) if section_exclude else None,
            )
        else:
            # Process all files
            sentence_tuples = parse_plain_folder_to_tuples(
                src,
                strict_metadata=strict_metadata,
                section_exclude=set(section_exclude) if section_exclude else None,
            )

        # Now process the (potentially sampled) tuples through spaCy
        from dm_annotations.io.loader import CorpusParser

        parser = CorpusParser(sentence_tuples, nlp)
        extractor = nlp.get_pipe("dm_extractor")
        docs = (extractor(doc) for doc in parser.stream())
        matches = (doc._.dm_matches for doc in docs)
    else:
        # Handle JSONL file - sample at the raw data level
        if sample < 1.0:
            import random
            from tempfile import NamedTemporaryFile

            with NamedTemporaryFile(
                mode="wb", suffix=".jsonl", delete=False
            ) as temp_file:
                with open(src, "rb") as f:
                    for line in f:
                        if random.random() < sample:
                            temp_file.write(line)
                temp_jsonl_path = Path(temp_file.name)
            extractor = nlp.get_pipe("dm_extractor")
            docs = (
                extractor(doc)
                for doc in parse_docs(
                    temp_jsonl_path, nlp, cache_batch_size, pipe_batch_size
                )
            )
            matches = (doc._.dm_matches for doc in docs)
            import atexit

            atexit.register(lambda: temp_jsonl_path.unlink(missing_ok=True))
        else:
            extractor = nlp.get_pipe("dm_extractor")
            docs = (
                extractor(doc)
                for doc in parse_docs(src, nlp, cache_batch_size, pipe_batch_size)
            )
            matches = (doc._.dm_matches for doc in docs)

    export_dms(matches, str(out_csv))
    # Only run export_count if CSV has data rows
    import os

    if os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
        import csv

        with open(out_csv, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            # skip header row
            try:
                next(reader)
            except StopIteration:
                pass
            else:
                if any(row for row in reader):
                    export_count(str(out_csv))


@app.command(
    "surfaces", help="(2/2) Generate per-pattern surface-form CSVs from a DM CSV."
)
def surfaces(
    dm_csv: Path = typer.Argument(..., help="CSV produced by `extract`"),
    out_dir: Path = typer.Argument(..., help="Output directory for per-pattern CSVs"),
    corpus: Path = typer.Option(
        None, "--corpus", "-c", help="Optional JSONL corpus for sampling"
    ),
    sample: bool = typer.Option(
        False, "--sample", "-s", help="Include up to 5 sample sentences"
    ),
    verbose: int = typer.Option(
        0,
        "--verbose",
        "-v",
        count=True,
        help="Increase verbosity (-v for INFO, -vv for DEBUG)",
    ),
) -> None:
    """Given a DM CSV, export per-pattern surface-form reports."""
    setup_logging(verbose)
    export_surface_forms(dm_csv, out_dir, corpus_jsonl=corpus, include_samples=sample)


@app.command("count", help="Compute and export frequency counts from a DM CSV.")
def count(
    dm_csv: Path = typer.Argument(..., help="DM‐CSV file from `extract`"),
    verbose: int = typer.Option(
        0,
        "--verbose",
        "-v",
        count=True,
        help="Increase verbosity (-v for INFO, -vv for DEBUG)",
    ),
) -> None:
    """Read an existing DM CSV and export `<stem>-counts.csv` and `.xlsx`."""
    setup_logging(verbose)
    export_count(str(dm_csv))


if __name__ == "__main__":
    app()
