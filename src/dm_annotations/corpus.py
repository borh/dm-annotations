import spacy
from spacy import displacy
from .matcher import modality_match, connectives_match
from pathlib import Path
import re
import unicodedata
import pandas as pd
import polars as pl

with open("jiyu.md") as f:
    text = f.read()

with open("ryu.md") as f:
    text = f.read()

colors = [
    "#F8766D",
    "#CD9600",
    "#7CAE00",
    "#00BE67",
    "#00BFC4",
    "#00A9FF",
    "#C77CFF",
    "#FF61CC",
]


def normalize_nfkc(s):
    if isinstance(s, str):
        return unicodedata.normalize("NFKC", s).replace(" ", "")
    return s


def read_annotations_excel(path):
    df = pd.read_excel(
        path,
    )
    # Apply NFKC normalization on string columns
    string_columns = df.select_dtypes(include=["object"]).columns
    df[string_columns] = df[string_columns].map(normalize_nfkc)

    column_patterns = [
        (r".*節の名称.*", "section_name"),
        (r".*文番号.*", "rid"),
        (r".*文構造.*", "dm"),
    ]
    rename_dict = {}
    for col in df.columns:
        for pattern, replacement in column_patterns:
            if re.search(pattern, col):
                rename_dict[col] = re.sub(pattern, replacement, col)
    df.rename(columns=rename_dict, inplace=True)
    return df[[column_name for _, column_name in column_patterns]]


if __name__ == "__main__":
    nlp = spacy.load("ja_core_news_trf")
    docs = nlp.pipe(text.splitlines())
    Path("ryu.html").unlink(missing_ok=True)
    Path("ryu_deps.html").unlink(missing_ok=True)
    for doc in docs:
        m_spans = modality_match(doc)
        c_spans = connectives_match(doc)

        doc.spans["modality"] = m_spans
        doc.spans["connectives"] = c_spans
        doc.spans["sc"] = m_spans + c_spans

        color_map = {span.label_: colors[0] for span in m_spans}
        color_map |= {span.label_: colors[1] for span in c_spans}

        svg = displacy.render(doc, style="span", options={"colors": color_map})
        deps_svg = displacy.render(doc, style="dep", options={"compact": True})
        Path("ryu.html").open("a").write(svg)
        Path("ryu_deps.html").open("a").write(deps_svg)
    for excel in Path("resources/analyses/").glob("*.xlsx"):
        assert read_annotations_excel(excel) is not None
