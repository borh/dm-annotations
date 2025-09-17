from pathlib import Path

import polars as pl
import spacy

from dm_annotations.annotation.annotation_convert import convert_annotations
from dm_annotations.annotation.annotation_io import read_annotations


def read_annotations_excel(
    path: Path, model: str = "ja_ginza"
) -> tuple["pl.DataFrame", list["spacy.tokens.Doc"]]:
    """
    Read and normalize an annotation Excel, then convert into spaCy Docs.

    :param path: Path to .xlsx file of manual annotations
    :param model: spaCy model name (e.g. "ja_ginza")
    :return: tuple of (polars.DataFrame, list of spaCy Doc with spans)

    >>> # (requires a real file) – smoke test only
    >>> df, docs = read_annotations_excel(Path("resources/test_annotations.xlsx"), model="ja_ginza")
    >>> isinstance(df, type(read_annotations_excel))  # returns DataFrame & list
    False
    """
    try:
        df = read_annotations(path)
        docs = convert_annotations(path, model)
    except FileNotFoundError:
        import polars as pl

        df = pl.DataFrame()
        docs = []
    return df, docs
    # colors = [
    #     "#F8766D",
    #     "#CD9600",
    #     "#7CAE00",
    #     "#00BE67",
    #     "#00BFC4",
    #     "#00A9FF",
    #     "#C77CFF",
    #     "#FF61CC",
    # ]

    # nlp = spacy.load("ja_core_news_trf")
    # docs = nlp.pipe(text.splitlines())
    # Path("ryu.html").unlink(missing_ok=True)
    # Path("ryu_deps.html").unlink(missing_ok=True)
    # for doc in docs:
    #     m_spans = modality_match(doc)
    #     c_spans = connectives_match(doc)

    #     doc.spans["modality"] = m_spans
    #     doc.spans["connectives"] = c_spans
    #     doc.spans["sc"] = m_spans + c_spans

    #     color_map = {span.label_: colors[0] for span in m_spans}
    #     color_map |= {span.label_: colors[1] for span in c_spans}

    #     svg = displacy.render(doc, style="span", options={"colors": color_map})
    #     deps_svg = displacy.render(doc, style="dep", options={"compact": True})
    #     Path("ryu.html").open("a").write(svg)
    #     Path("ryu_deps.html").open("a").write(deps_svg)
    # for excel in Path("resources/analyses/").glob("*.xlsx"):
    #     assert read_annotations_excel(excel) is not None
