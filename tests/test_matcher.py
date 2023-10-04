import pytest
import spacy
from dm_annotations.matcher import connectives_match, modality_match


@pytest.fixture
def doc():
    nlp = spacy.load("ja_core_news_sm")
    doc = nlp(
        """後は，実験をしなければいけないのだろう。
    が，そのようなことがあるが。
    このことから，ないほうがよいでしょう。
    その結果，施策としてコミュニティ･バスによりフォーカスした群で公共交通に対する態度･行動変容効果が示唆された一方，相対的に自動車利用抑制にフォーカスした群においては，自動車利用抑制に対する態度･行動変容効果が見られ，本研究の仮説が支持されたことが示唆された．"""
    )
    return doc


def test_connectives_match(doc):
    matches = connectives_match(doc)
    assert len(matches) > 0


def test_modality_match(doc):
    matches = modality_match(doc)
    assert len(matches) > 0
