from logging import warn
from pathlib import Path

from dm_annotations.annotation_loader import read_annotations_excel


def test_read_annotations_excel(model):
    for excel in Path("resources/analyses/").glob("*.xlsx"):
        print(excel)
        warn(excel)
        df, docs = read_annotations_excel(excel, model=model)
        assert len(df) > 0
        assert len(docs) > 0

        print(df["connective_meidai_check"].count())
        print(df["modality_meidai_check"].count())

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
