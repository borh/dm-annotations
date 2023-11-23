from pathlib import Path
from logging import warn
from dm_annotations.corpus import *


def test_corpus_load():
    with open("jiyu.md") as f:
        text = f.read()
    assert len(text) > 0

    with open("ryu.md") as f:
        text = f.read()
    assert len(text) > 0


def test_read_annotations_excel():
    for excel in Path("resources/analyses/").glob("*.xlsx"):
        warn(excel)
        df, docs = read_annotations_excel(excel)
        assert len(df) > 0
        assert len(docs) > 0

        assert {"section_name", "rid", "dm"} <= set(df.columns.to_list())
        # [
        #     "項目番号",
        #     "章節の名称",
        #     "文段番号",
        #     "文番号",
        #     "segment数",
        #     "テクスト単位（文単位）",
        #     "Segment別テキスト（下線部は命題）",
        #     "文構造",
        #     "文段番号.1",
        #     "文番号.1",
        #     "文頭句",
        #     "接続表現",
        #     "命題内(0)か外(1)か",
        #     "備考",
        #     "語連鎖：文末表現（紫）",
        #     "文末表現意味機能分類",
        #     "備考１(定義）",
        #     "備考２（構文構造の説明）",
        #     "Unnamed: 18",
        #     "Unnamed: 19",
        # ]

        # assert len(df.groupby("section_name")) > 1
