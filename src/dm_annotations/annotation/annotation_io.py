"""
I/O for annotation spreadsheets: use Polars to read & normalize.
"""

from pathlib import Path

import jaconv
import polars as pl


def _normalize(s: str) -> str:
    return jaconv.h2z(
        jaconv.normalize(s.replace(" ", ""), "NFKC"),
        digit=False,
        ascii=True,
        kana=True,
    ).rstrip("　")


def read_annotations(path: Path) -> pl.DataFrame:
    """
    Read an Excel file into a DataFrame, applying NFKC normalization.
    """
    df_pl = pl.read_excel(path)
    for col in df_pl.select(pl.Utf8).columns:  # type: ignore[attr-defined]
        df_pl = df_pl.with_columns(pl.col(col).apply(_normalize).alias(col))  # type: ignore[attr-defined]
    return df_pl
